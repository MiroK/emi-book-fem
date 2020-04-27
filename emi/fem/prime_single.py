# Primal, singlescale
from dolfin import *
from xii import *
from emi.utils import H1_norm, L2_norm
import emi.fem.common as common
from xii.assembler.trace_matrix import trace_mat_no_restrict


# Dropping the multiplier and still having a symmetric problem
def setup_problem(n, mms, params):
    '''Single-dimensional primal formulation'''
    base_mesh = UnitSquareMesh(mpi_comm_self(), *(n, )*2)

    # Marking
    inside = ['(0.25-tol<x[0])', '(x[0] < 0.75+tol)', '(0.25-tol<x[1])', '(x[1] < 0.75+tol)']
    inside = CompiledSubDomain(' && '.join(inside), tol=1E-10)

    mesh_f = MeshFunction('size_t', base_mesh, base_mesh.topology().dim(), 0)
    inside.mark(mesh_f, 1)

    inner_mesh = SubMesh(base_mesh, mesh_f, 1)  # Inside
    outer_mesh = SubMesh(base_mesh, mesh_f, 0)  # Ouside
    interface_mesh = BoundaryMesh(inner_mesh, 'exterior')
            
    # Spaces
    V0 = FunctionSpace(outer_mesh, 'CG', 1)
    V1 = FunctionSpace(inner_mesh, 'CG', 1)
    
    W = [V0, V1]

    u0, u1 = map(TrialFunction, W)
    v0, v1 = map(TestFunction, W)

    Tu0, Tv0 = (Trace(f, interface_mesh) for f in (u0, v0))
    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))

    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    # The line integral
    n = OuterNormal(interface_mesh, [0.5, 0.5])
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    kappa, epsilon = map(Constant, (params.kappa, params.eps))
    
    a = block_form(W, 2)

    a[0][0] = kappa*inner(grad(u0), grad(v0))*dx + (1./epsilon)*inner(Tu0, Tv0)*dx_
    a[0][1] = -(1./epsilon)*inner(Tu1, Tv0)*dx_

    a[1][0] = -(1./epsilon)*inner(Tu0, Tv1)*dx_
    a[1][1] = inner(grad(u1), grad(v1))*dx + (1./epsilon)*inner(Tu1, Tv1)*dx_

    # Data
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    L = block_form(W, 1)
    L[0] = inner(f1, v0)*dx
    L[0] += sum((1./epsilon)*inner(gi, Tv0)*dx_(i) for i, gi in enumerate(gGamma, 1))
     
    # Iface contribution
    L[1] = inner(f, v1)*dx
    L[1] += -sum((1./epsilon)*inner(gi, Tv1)*dx_(i) for i, gi in enumerate(gGamma, 1))
    
    A, b = map(ii_assemble, (a, L))
    
    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    V0_bcs = [DirichletBC(V0, gi, facet_f, i) for i, gi in enumerate(gBdry, 1)]
    bcs = [V0_bcs, []]

    A, b = apply_bc(A, b, bcs)

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    # Error of the solution ...
    exact = mms_data.solution
    subdomains = mms_data.subdomains[1]
    
    def get_error(wh, subdomains=subdomains, exact=exact):
        u1h, uh = wh

        sigma_exact, u_exact, p_exact, I_exact = exact

        V = uh.function_space()
        mesh = V.mesh()
        #
        # Recover the transmembrane potential by postprocessing
        #
        V = uh.function_space()
        mesh = V.mesh()
        interface_mesh = BoundaryMesh(mesh, 'exterior')
        
        V = FunctionSpace(interface_mesh, 'CG', 1)
        Tu1 = PETScMatrix(trace_mat_no_restrict(u1h.function_space(), V))*u1h.vector()
        Tu = PETScMatrix(trace_mat_no_restrict(uh.function_space(), V))*uh.vector()
        
        vh = Function(V, Tu1 - Tu)
        # On finer
        #V = FunctionSpace(interface_mesh, 'CG', 3)
        # Hs = matrix_fromHs(HsNorm(V, s=0.5))
        ## Use interpolate
        #vh = interpolate(vh, V)
            
        return (sqrt(H1_norm(u_exact[0], u1h)**2 + H1_norm(u_exact[1], uh)**2),
                sqrt(L2_norm(u_exact[0], u1h)**2 + L2_norm(u_exact[1], uh)**2),
                #Aerror(Hs, V, p_exact, vh))
                L2_norm(p_exact, vh))
    
    error_types = ('|u|_1', '|u|_0', '|v|_0')
    
    return get_error, error_types

# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps')
