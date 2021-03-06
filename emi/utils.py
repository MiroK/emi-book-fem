from dolfin import (Function, LUSolver, FunctionSpace, errornorm, interpolate,
                    Expression, CompiledSubDomain, MeshFunction, SubMesh, sqrt)
from collections import namedtuple
from xii import ii_Function
import numpy.ma as mask
import numpy as np
import itertools
import ufl
import re


def serialize_mixed_space(space):
    '''Representation of space for cbc.block'''
    elm = space.ufl_element()
    sub_elms = elm.sub_elements()
    if not sub_elms:
        return space

    mesh = space.mesh()
    return [FunctionSpace(mesh, elm) for elm in sub_elms]


def direct_solve(A, b, W, which='umfpack'):
    '''inv(A)*b'''
    print 'Solving system of size %d' % A.size(0)
    # NOTE: umfpack sometimes blows up, MUMPS produces crap more often than not
    if isinstance(W, list):
        wh = ii_Function(W)
        LUSolver(which).solve(A, wh.vector(), b)
        print('|b-Ax| from direct solver', (A*wh.vector()-b).norm('linf'))
        
        return wh
    
    wh = Function(W)
    LUSolver(which).solve(A, wh.vector(), b)
    print('|b-Ax| from direct solver', (A*wh.vector()-b).norm('linf'))

    if isinstance(W.ufl_element(), (ufl.VectorElement, ufl.TensorElement)) or W.num_sub_spaces() == 1:
        return ii_Function([W], [wh])

    # Now get components
    Wblock = serialize_mixed_space(W)
    wh = wh.split(deepcopy=True)
    
    return ii_Function(Wblock, wh)


def get_problem_parameters(arguments, params):
    '''Look for parameter values in the commandline arguments'''
    # Keywords
    words = tuple(('-param_%s' % param) for param in params)
    # For convenience if they are missing each get a value 1
    iterators = {}  # param -> values that it will take
    expressions = {}  # param -> how to compute the value
    if not any(word in arg for arg in arguments for word in words):
        for p in params:
            iterators[p] = (1, )
        # And all is petsc arguments
    # Values for the parameter are numbers folowing the word or expression
    # which makes it dependent on non-number
    else:
        number = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')

        is_number = lambda w: number.match(w) is not None
    
        for param, word in zip(params, words):
            start = stop = arguments.index(word)+1
            while stop < len(arguments) and is_number(arguments[stop]):
                stop += 1

            # Numbers
            if start < stop:
                try:
                    param_values = map(float, arguments[start:stop])
                except ValueError:
                    param_values = map(eval, arguments[start:stop])
                iterators[param] = param_values
            # Okay now there will be some work to do to get iteration
            else:
                # Try to get the expression for computing value of the parameter
                # e.g -param_mu "lambda x: x+1"
                expr = arguments[start:start+1][0]
                expressions[param] = eval(expr)
            # Chop off parameter values - the remainig guys belong to petsc
            # NOTE: Start - 1 to include the keyword
            arguments = arguments[:start-1] + arguments[stop:]

    # As part of validity check there should be not more -param then needed
    assert not any(arg.startswith('-param_') for arg in arguments), 'Pass params only for %s' % params

    ParameterSetBase = namedtuple('p', params)

    class ParameterSet(ParameterSetBase):
        def __init__(self, *args, **kwargs):
            # assert all(v >= 0 for v in args), 'Problem parameters must be positive'
            # assert all(v >= 0 for v in kwargs.values()), 'Problem parameters must be positive'
            ParameterSetBase.__init__(self, *args, **kwargs)

    # For numbers
    if not expressions:
        params, iterators = iterators.keys(), iterators.values()
        parameters = (ParameterSet(**dict(zip(params, p))) for p in itertools.product(*iterators))
    # There will be an expression giving value of all the parameters in terms
    # of those that have numerical range
    else:
        get_expr = parameter_expression(expressions, params)
        # Get expr takes in parameters with given numerical ranges and
        # return a dict where are the parameters are present
        params, iterators = iterators.keys(), iterators.values()
        
        parameters = (ParameterSet(**get_expr(**dict(zip(params, p))))
                      for p in itertools.product(*iterators))
    # These guys come in pair key: value
    petsc_arguments = dict(zip(arguments[0::2], arguments[1::2]))

    return parameters, petsc_arguments


def split_jobs(comm, jobs):
    '''Divide parameter sweeping'''
    jobs = list(jobs)

    njobs = len(jobs)
    # comm (rank/nprocs)
    if isinstance(comm, str):
        if not comm:
            rank, nprocs = 0, 1
        # Parse
        else:
            rank, nprocs = map(int, comm.strip().split('/'))
    else:
        nprocs = comm.tompi4py().size
        rank = comm.tompi4py().rank
        
    assert 0 <= rank < nprocs

    size = njobs/nprocs

    first = rank*size
    last = njobs if rank == nprocs-1 else (rank+1)*size

    my_jobs = [jobs[i] for i in range(first, last)]
    assert my_jobs

    return my_jobs


H1_norm = lambda u, uh, degree_rise=2: errornorm(u, uh, 'H1', degree_rise=degree_rise)

Hdiv_norm = lambda u, uh, degree_rise=2: errornorm(u, uh, 'Hdiv', degree_rise=degree_rise)

L2_norm = lambda u, uh, degree_rise=2: errornorm(u, uh, 'L2', degree_rise=degree_rise)


def subdomain_interpolate(pairs, V, reduce='last'):
    '''(f, chi), V -> Function fh in V such that fh|chi is f'''
    array = lambda f: f.vector().get_local()
    fs, chis = zip(*pairs)
    
    fs = np.column_stack([array(interpolate(f, V)) for f in fs])
    x = np.column_stack([array(interpolate(Expression('x[%d]' % i, degree=1), V)) for i in range(V.mesh().geometry().dim())])

    chis = [CompiledSubDomain(chi, tol=1E-10) for chi in chis]
    is_masked = np.column_stack([map(lambda xi, chi=chi: not chi.inside(xi, False), x) for chi in chis])

    fs = mask.masked_array(fs, is_masked)

    if reduce == 'avg':
        values = np.mean(fs, axis=1)
    elif reduce == 'max':
        values = np.choose(np.argmax(fs, axis=1), fs.T)
    elif reduce == 'min':
        values = np.choose(np.argmin(fs, axis=1), fs.T)
    elif reduce == 'first':
        choice = [row.tolist().index(False) for row in is_masked]
        values = np.choose(choice, fs.T)
    elif reduce == 'last':
        choice = np.array([row[::-1].tolist().index(False) for row in is_masked], dtype=int)
        nsubs = len(chis)
        values = np.choose(nsubs-1-choice, fs.T)        
    else:
        raise ValueError

    fh = Function(V)
    fh.vector().set_local(values)

    return fh


def broken_norm(norm, subdomains, mesh=None):
    '''Eval norm on subdomains and combine'''
    # Subdomains can be string -> get compile
    if isinstance(first(subdomains), str):
        subdomains = [CompiledSubDomain(s, tol=1E-10) for s in subdomains]
        mesh = None

        def get_norm(u, uh):
            assert len(subdomains) == len(u)
    
            V = uh.function_space()
            mesh = V.mesh()
            cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

            error = 0
            for subd_i, u_i in zip(subdomains, u):
                cell_f.set_all(0)  # Reset!
                # Pick an edge
                subd_i.mark(cell_f, 1)
                mesh_i = SubMesh(mesh, cell_f, 1)
                # Edge local function space
                Vi = FunctionSpace(mesh_i, V.ufl_element())
                # The solution on it
                uh_i = interpolate(uh, Vi)
                # And the error there
                error_i = norm(u_i, uh_i)
                error += error_i**2
            error = sqrt(error)
            return error
        # Done
        return get_norm

    # CompiledSubDomain -> will be used to mark (go to base case)
    if hasattr(first(subdomains), 'mark'):
        return broken_norm(norm, subdomains, mesh=mesh)
    
    # Tuples of (tag, mesh_function)
    # Do some consistency checks; same mesh
    _, = set(first(p).mesh().id() for p in subdomains)
    mesh = first(first(subdomains)).mesh()
    # Cell functions ?
    _, = set(first(p).dim() for p in subdomains)
    dim = first(first(subdomains)).dim()

    assert mesh.topology().dim() == dim

    def get_norm(u, uh, mesh=mesh):
        assert len(subdomains) == len(u)
        
        V = uh.function_space()
        assert mesh.id() == V.mesh().id()

        error = 0
        # NOTE: u is tag -> solution
        for subd, tag in subdomains:
            mesh_i = SubMesh(mesh, subd, tag)
            # Edge local function space
            Vi = FunctionSpace(mesh_i, V.ufl_element())
            # The solution on it
            uh_i = interpolate(uh, Vi)
            # And the error there
            error_i = norm(u[tag], uh_i)
            error += error_i**2
        error = sqrt(error)
        return error
    # Done
    return get_norm


# First of anything (non-empty)
def first(iterable): return next(iter(iterable))
