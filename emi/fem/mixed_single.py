# Mixed, singlescale
from dolfin import *
from xii import *
from emi.utils import Hdiv_norm, L2_norm, broken_norm
import emi.fem.common as common


def setup_problem(n, mms, params):
    '''Single-dimensional mixed formulation'''
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)

    # Marking of intra/extra-cellular domains
    outside, inside = mms.subdomains[2]

    cell_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    CompiledSubDomain(outside, tol=1E-10).mark(cell_f, 0)  # Not needed
    CompiledSubDomain(inside, tol=1E-10).mark(cell_f, 1)

    # These are just auxiliary so that interface can be grabbed
    inner_mesh = SubMesh(base_mesh, cell_f, 1)  # Inside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
    
    # Spaces
    V1 = FunctionSpace(base_mesh, 'RT', 1)
    Q1 = FunctionSpace(base_mesh, 'DG', 0)

    W = [V1, Q1]

    sigma1, u1 = map(TrialFunction, W)
    tau1, v1 = map(TestFunction, W)

    # Hdiv trace should you normal (though orientation seems not important)
    n = OuterNormal(interface_mesh, [0.5, 0.5]) 

    Tsigma1, Ttau1 = (Trace(f, interface_mesh, '+', n) for f in (sigma1, tau1))

    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    dx = Measure('dx', domain=base_mesh, subdomain_data=cell_f)
    # The line integral
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    kappa1, epsilon = map(Constant, (params.kappa, params.eps))
    
    a = block_form(W, 2)
    a[0][0] = inner((1./kappa1)*sigma1, tau1)*dx(0) + inner(sigma1, tau1)*dx(1)
    a[0][0] += epsilon*inner(dot(Tsigma1, n), dot(Ttau1, n))*dx_
    a[0][1] = inner(div(tau1), u1)*dx
    a[1][0] = inner(div(sigma1), v1)*dx
    
    # Data
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    L = block_form(W, 1)

    # Outer boundary contribution
    n1 = FacetNormal(base_mesh)
    # Piece by piece
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    ds = Measure('ds', domain=base_mesh, subdomain_data=facet_f)
    L[0] = sum(inner(gi, dot(Ttau1, n1))*ds(i) for i, gi in enumerate(gBdry, 1))
    # Iface contribution
    L[0] += -sum(inner(gi, dot(Ttau1, n))*dx_(i) for i, gi in enumerate(gGamma, 1))
    
    L[1] = -inner(f1, v1)*dx(0) - inner(f, v1)*dx(1)

    A, b = map(ii_assemble, (a, L))

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    exact = mms_data.solution
    ifaces = mms_data.subdomains[1]
    subdomains = mms_data.subdomains[2]

    def get_error(wh, subdomains=subdomains, ifaces=ifaces, exact=exact,
                  normals=mms_data.normals[0], params=params, mms=mms_data):
        sigmah, uh = wh
        sigma_exact, u_exact, p_exact, I_exact = exact
        #########################################
        # Get the difference by postprocessing
        #########################################
        base_mesh = sigmah.function_space().mesh()
        
        cell_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
        CompiledSubDomain(subdomains[0], tol=1E-10).mark(cell_f, 0)  # Not needed
        CompiledSubDomain(subdomains[1], tol=1E-10).mark(cell_f, 1)

        # These are just auxiliary so that interface can be grabbed
        inner_mesh = SubMesh(base_mesh, cell_f, 1)  # Inside
        interface_mesh = BoundaryMesh(inner_mesh, 'exterior')

        n = OuterNormal(interface_mesh, [0.5, 0.5])
        dx_ = Measure('dx', domain=interface_mesh)
        
        Tu1 = Trace(uh, interface_mesh, '+', n)
        Tu2 = Trace(uh, interface_mesh, '-', n)

        Q = FunctionSpace(interface_mesh, 'DG', 0)
        p, q = TrialFunction(Q), TestFunction(Q)
        
        a = inner(p, q)*dx
        L = inner(Tu1 - Tu2, q)*dx_
        ph = Function(Q)
        
        A, b = map(ii_convert, map(ii_assemble, (a, L)))
        # L^2 projection
        x = as_backend_type(ph.vector()).vec()
        A = as_backend_type(A).mat().getDiagonal()
        b =  as_backend_type(b).vec()
        x.pointwiseDivide(b, A)

        marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
        [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, ifaces), 1)]
        # The line integral
        dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

        eps = Constant(params.eps)
        L = (sum(inner(gi, q)*dx_(i) for i, gi in enumerate(mms.rhs[3], 1)))
        L += eps*inner(dot(Trace(sigmah, interface_mesh, '+', n), n), q)*dx_
        
        b = ii_convert(ii_assemble(L))
        ph_E = Function(Q)
        x = as_backend_type(ph_E.vector()).vec()
        b =  as_backend_type(b).vec()
        x.pointwiseDivide(b, A)
        
        return (broken_norm(Hdiv_norm, subdomains[:])(sigma_exact[:], sigmah),
                broken_norm(L2_norm, subdomains[:])(u_exact[:], uh),
                L2_norm(p_exact, ph),
                L2_norm(p_exact, ph_E))

    error_types = ('|sigma|_div', '|u|_0', '|v|_{0I}', '|v|_{0E}')
    
    return get_error, error_types
                
# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps')
