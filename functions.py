# The script containing auxiliary generic functions

import os, sys, time, signal

def error(message, specification = ''):
    """Function for error reporting"""
    sys.stderr.write('Error{}: {}\n'.format(specification, message))
    sys.stderr.flush()
    time.sleep(1)
    parent = os.popen('ps -p {}'.format(os.getppid())).read().split()[-1]
    gparent_pid = os.popen("ps -p {} -oppid=".format(os.getppid())).read().\
    strip()
    ##
    gparent = os.popen('ps -p {}'.format(gparent_pid)).read().split()[-1]
    if parent == 'mpiexec':
        from mpi4py import MPI
        MPI.COMM_WORLD.Abort(1)
    elif gparent == 'mpiexec':
        os.kill(os.getppid(), signal.SIGTERM)
    else:
        sys.exit(1)