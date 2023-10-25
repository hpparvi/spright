from numpy import abs, array, inf, isfinite
from scipy.stats import norm

def is_puffy_population_ok(pv, rdm):
    d = pv[1] - pv[0]
    a = 0.5 - abs(pv[2] - 0.5)
    r3 = pv[0] + d * (pv[2] + pv[3] * a)
    puffy_rho_at_r3 = pv[6] * pv[0] ** pv[7] / 2.0 ** pv[7]
    rocky_rho_at_r3 = rdm.evaluate_rocky(0.0, array([r3]))[0]
    if isfinite(rocky_rho_at_r3) and (puffy_rho_at_r3 > rocky_rho_at_r3):
        return False
    else:
        return True


pcw = norm(0.5, 0.1)
psp = norm(-0.5, 1.5)
pssr = norm(0.0, 0.35)
pssw = norm(0.0, 0.35)
pssp = norm(0.0, 0.35)

h0 = array([[ 0.8,  2.1],
            [ 1.8,  3.2],
            [ 0.0,  1.0],
            [-1.0,  1.0],
            [ 0.1,  0.4],
            [ 0.1,  1.0],
            [ 2.0,  5.0],
            [-2.5,  0.1],
            [-1.2,  0.5],
            [-1.2,  1.0],
            [-1.2,  0.5]])


def transform_base(u, wslow=0.0, wshigh=1.0, water_model='z19'):
    tr = h0[:,0] + u*(h0[:,1]-h0[:,0])
    tr[2]  = wslow + u[2] * (wshigh - wslow)
    if water_model == 'z19':
        tr[5]  = pcw.ppf(u[5])
    tr[7]  = psp.ppf(u[7])
    tr[8]  = pssr.ppf(u[8])
    tr[9]  = pssw.ppf(u[9])
    tr[10] = pssp.ppf(u[10])
    return tr
