# Primal, multiscale
from dolfin import *
from xii import *
from emi.utils import H1_norm, L2_norm, subdomain_interpolate
import emi.fem.common as common
from xii.assembler.trace_matrix import trace_mat_no_restrict



def setup_problem(n, mms, params):
    '''Multi-dimensional primal formulation'''
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
    V1 = FunctionSpace(outer_mesh, 'CG', 1)
    V = FunctionSpace(inner_mesh, 'CG', 1)
    Q = FunctionSpace(interface_mesh, 'CG', 1)
    
    W = [V1, V, Q]

    u1, u, p = map(TrialFunction, W)
    v1, v, q = map(TestFunction, W)

    Tu1, Tv1 = (Trace(f, interface_mesh) for f in (u1, v1))
    Tu, Tv = (Trace(f, interface_mesh) for f in (u, v))

    # Mark subdomains of the interface mesh (to get source terms therein)
    subdomains = mms.subdomains[1]  # 
    marking_f = MeshFunction('size_t', interface_mesh, interface_mesh.topology().dim(), 0)
    [subd.mark(marking_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    # The line integral
    dx_ = Measure('dx', domain=interface_mesh, subdomain_data=marking_f)

    kappa, epsilon = map(Constant, (params.kappa, params.eps))
    
    a = block_form(W, 2)

    a[0][0] = kappa*inner(grad(u1), grad(v1))*dx
    a[0][2] = inner(Tv1, p)*dx_

    a[1][1] = inner(grad(u), grad(v))*dx
    a[1][2] = -inner(Tv, p)*dx_

    a[2][0] = inner(Tu1, q)*dx_
    a[2][1] = -inner(Tu, q)*dx_
    a[2][2] = -epsilon*inner(p, q)*dx_

    # Data
    # Source for domains, outer boundary data, source for interface
    f1, f, gBdry, gGamma, hGamma = mms.rhs

    L = block_form(W, 1)
    L[0] = inner(f1, v1)*dx
    L[0] += sum(inner(hi, Tv1)*dx_(i) for i, hi in enumerate(hGamma, 1))
    
    # Iface contribution
    L[1] = inner(f, v)*dx
    L[2] = sum(inner(gi, q)*dx_(i) for i, gi in enumerate(gGamma, 1))
    
    A, b = map(ii_assemble, (a, L))

    subdomains = mms.subdomains[0]  # 
    facet_f = MeshFunction('size_t', outer_mesh, outer_mesh.topology().dim()-1, 0)
    [subd.mark(facet_f, i) for i, subd in enumerate(map(CompiledSubDomain, subdomains), 1)]

    V1_bcs = [DirichletBC(V1, gi, facet_f, i) for i, gi in enumerate(gBdry, 1)]
    bcs = [V1_bcs, [], []]

    A, b = apply_bc(A, b, bcs)

    return A, b, W


setup_mms = common.setup_mms


def setup_error_monitor(mms_data, params):
    '''Compute function mapping numerical solution to errors'''
    # Error of the solution ...
    
    exact = mms_data.solution
    subdomains = mms_data.subdomains[1]
    
    def get_error(wh, subdomains=subdomains, exact=exact, mms=mms_data, params=params):
        u1h, uh, ph = wh
        sigma_exact, u_exact, p_exact, I_exact = exact

        # Mutliplier error
        mesh = ph.function_space().mesh()
        Q = FunctionSpace(mesh, 'CG', 3)

        cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        # Spaces on pieces
        for tag, subd in enumerate(subdomains, 1):
            CompiledSubDomain(subd, tol=1E-10).mark(cell_f, tag)
        dx = Measure('dx', domain=mesh, subdomain_data=cell_f)

        p, q = TrialFunction(Q), TestFunction(Q)
        a = inner(p, q)*dx
        L = sum(inner(I_exact_i, q)*dx(i) for i, I_exact_i in enumerate(I_exact, 1))
        I_exact = Function(Q)
        A, b = map(assemble, (a, L))
        solve(A, I_exact.vector(), b)

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

        # Now using flux we have
        Q = ph.function_space()
        mesh = Q.mesh()
        
        cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        # Spaces on pieces
        for tag, subd in enumerate(subdomains, 1):
            CompiledSubDomain(subd, tol=1E-10).mark(cell_f, tag)
        dx = Measure('dx', domain=mesh, subdomain_data=cell_f)

        p, q = TrialFunction(Q), TestFunction(Q)
        gGamma = mms.rhs[3]
        
        a = inner(p, q)*dx
        L = inner(params.eps*ph, q)*dx + sum(inner(gi, q)*dx(i) for i, gi in enumerate(gGamma, 1))
        A, b = map(assemble, (a, L))
        vh_P = Function(Q)  # Since we project
        solve(A, vh_P.vector(), b)

        vh_I = subdomain_interpolate(zip(gGamma, subdomains), Q)
        vh_I.vector().axpy(params.eps, ph.vector())

        # Simply by interpolation

        return (sqrt(H1_norm(u_exact[0], u1h)**2 + H1_norm(u_exact[1], uh)**2),
                L2_norm(p_exact, vh),
                L2_norm(p_exact, vh_P),
                L2_norm(p_exact, vh_I))
    
    error_types = ('|u|_1', '|v|_0', '|v|_{0P}', '|v|_{0I}')
    
    return get_error, error_types

# --------------------------------------------------------------------

# How is the problem parametrized
PARAMETERS = ('kappa', 'eps')
