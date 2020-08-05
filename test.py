import numpy as np # (riepybdlib requires numpy arrays)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %matplotlib inline
from matplotlib import pyplot as plt
import riepybdlib.manifold as rm   # For the manifold functionality
import riepybdlib.statistics as rs # For statistics on riemannian manifolds
import riepybdlib.data as pbddata  # Some data to showcase the toolbox
import riepybdlib.plot as pbdplt   # Plot functionality (relies on matplotlib)
import riepybdlib.s2_fcts as s2_fcts

cols = cm.Set1(np.linspace(0,1,10))[:,0:3] # Standard plot colors
m_3deucl = rm.get_euclidean_manifold(3,'3d-eucl')
m_time = rm.get_euclidean_manifold(1,'time')
m_quat = rm.get_quaternion_manifold('orientation')
m_s2   = rm.get_s2_manifold('2-sphere')


m_dem = m_time*m_3deucl*m_quat
m_times2 = m_time*m_s2
print(m_dem.name)
print(m_times2.name)
print(m_times2.n_dimM)
print(m_times2.n_dimT)
mu = m_times2.id_elem
print(mu)
sigma = np.eye(m_times2.n_dimT)

gauss = rs.Gaussian(m_times2, mu,sigma)
te  = pbddata.get_letter_dataS2(letter='I',n_samples=4,use_time=False)
m_data = np.vstack(pbddata.get_letter_dataS2(letter='I',n_samples=4,use_time=False))

g = rs.Gaussian(m_s2).mle(m_data)
plt.figure(figsize=(5,5) )
ax = plt.subplot(111,projection='3d')
s2_fcts.plot_manifold(ax) # Manifold
plt.plot(m_data[:,0],m_data[:,1],m_data[:,2],'.', label='data') # Original Data
pbdplt.plot_gaussian_s2(ax,g.mu,g.sigma,label='Gaussian')            # Gaussian
plt.legend();
plt.show()


# Load 4 demonstrations:
dems = pbddata.get_letter_dataS2(letter='I',n_samples=4,use_time=True)

# Combine seperate demonstrations in one list:
m_data= [point for dem in dems for point in dem]
print(m_data)

# Train Gaussian:

g = rs.Gaussian(m_times2).mle(m_data)

results = []
i_in  = 0 # Input manifold index
i_out = 1 # Output manifold index
for p in list(m_data[0:200:10]):
    results.append(g.condition(p[i_in],i_in=i_in,i_out=i_out))

plt.figure(figsize=(5,5) )
ax = plt.subplot(111,projection='3d')

s2_fcts.plot_manifold(ax) # Manifold
plt_data = m_times2.swapto_tupleoflist(m_data)
plt.plot(plt_data[1][:,0],plt_data[1][:,1],plt_data[1][:,2],'.', label='data')     # Original Data
s2_fcts.plot_gaussian(ax,g.margin(1).mu,g.margin(1).sigma,label='Gaussian')  # Gaussian
label = 'Condition result'
for r in results:
    s2_fcts.plot_gaussian(ax,r.mu,r.sigma,showtangent=False,
                         linealpha=0.3,color='yellow',label=label)
    label=''

plt.legend();
plt.show()
g1 = rs.Gaussian(m_s2, np.array([1,0,0]),np.diag([0.1,0.01]))
g2 = rs.Gaussian(m_s2, np.array([0,-1,0]),np.diag([0.1,0.01]))

g1g2= g1*g2

plt.figure(figsize=(5,5) )
ax = plt.subplot(111,projection='3d')

s2_fcts.plot_manifold(ax) # Manifold
s2_fcts.plot_gaussian(ax,g1.mu,g1.sigma,color=[1,0,0])
s2_fcts.plot_gaussian(ax,g2.mu,g2.sigma,color=[0,1,0])
s2_fcts.plot_gaussian(ax,g1g2.mu,g1g2.sigma,color=[1,1,0])
plt.show()

gmm = rs.GMM(m_times2,6)
# Get some data to train the GMM:
dems = pbddata.get_letter_dataS2(letter='S',n_samples=4,use_time=True)
m_data= [point for dem in dems for point in dem]

gmm.kmeans(m_data)             # Initialization
lik,avglik = gmm.fit(m_data)   # Expectation Maximiation
plt.figure(figsize=(4,2))
plt.plot(avglik)
plt.grid('on')
plt.xlabel('Iteration')
plt.ylabel(r'$-\frac{1}{N}\sum_i^N \ln(\mathcal{P}(x_i))$')
plt.show()


plt.figure(figsize=(5,5) )
ax = plt.subplot(111,projection='3d')
s2_fcts.plot_manifold(ax) # Manifold

# Data:
plt_data = m_times2.swapto_tupleoflist(m_data)
plt.plot(plt_data[1][:,0],plt_data[1][:,1],plt_data[1][:,2],
         '.', label='data',color='gray',alpha=0.5)

# GMM:
for i,g in enumerate(gmm.gaussians):
    gtmp = g.margin(1) # Compute margin to display only s2
    s2_fcts.plot_gaussian(ax,gtmp.mu,gtmp.sigma,color=cols[i,:])
plt.show()


i_in  = 0 # Input manifold index
i_out = 1 # Output manifold index
results = []
for p in list(m_data[0:200:1]):
    results.append(gmm.gmr(p[i_in],i_in=i_in,i_out=i_out)[0])

plt.figure(figsize=(5,5) )
ax = plt.subplot(111,projection='3d')

s2_fcts.plot_manifold(ax) # Manifold
plt_data = m_times2.swapto_tupleoflist(m_data)
plt.plot(plt_data[1][:,0],plt_data[1][:,1],plt_data[1][:,2],'.', label='data')     # Original Data
s2_fcts.plot_gaussian(ax,g.margin(1).mu,g.margin(1).sigma,label='Gaussian')  # Gaussian
label = 'Condition result'
for r in results:
    s2_fcts.plot_gaussian(ax,r.mu,r.sigma,showtangent=False,
                         linealpha=0.3,color='yellow',label=label)
    label=''

plt.legend()
plt.show()

(data, _, _) = pbddata.get_tp_dataS2(use_time=True)

# Construct data structure for training:
f_data = []
f_data.append( m_times2.swapto_tupleoflist([p for dem in data[0] for p in dem  ]) )
f_data.append( m_times2.swapto_tupleoflist([p for dem in data[1] for p in dem  ]) )
tpdata = (f_data[0][0], f_data[0][1], f_data[1][1])

mytpgmm = rs.GMM(m_time*m_s2*m_s2, n_components=3)
mytpgmm.init_time_based(tpdata[0], tpdata)
mytpgmm.fit(tpdata);

# The GMM in each frame can be computed by marginalizing:
gmms = []
gmms.append(mytpgmm.margin([0,1])) # Frame 1
gmms.append(mytpgmm.margin([0,2])) # Frame 2

plt.figure(figsize=(10, 5))

for i, gmm in enumerate(gmms):
    ax = plt.subplot(1, 2, i + 1, projection='3d')
    s2_fcts.plot_manifold(ax)

    # Plot data:
    for j, dem in enumerate(data[i]):
        s2data = m_times2.swapto_tupleoflist(dem)[1]
        plt.plot(s2data[:, 0], s2data[:, 1], s2data[:, 2], color=cols[j,])

    # Plot Gaussians
    for j, g in enumerate(gmm.gaussians):
        g = g.margin(1)
        s2_fcts.plot_gaussian(ax, g.mu, g.sigma, color=cols[i, :])
        ax.text(g.mu[0], g.mu[1], g.mu[2], '{0}'.format(j + 1))
    ax.view_init(30, 60)
plt.show()
th1 = 2 * np.pi / 6
ph1 = 4 * np.pi / 6
th2 = 3 * np.pi / 6
ph2 = 1 * np.pi / 6
bs = [(0, s2_fcts.sphere(1, th1, ph1)),  # New position of frame 1
      (0, s2_fcts.sphere(1, th2, ph2))]  # New position of frame 2
A1 = np.eye(3)
A1[1:, 1:] = s2_fcts.fR(-1 * np.pi / 3)
A2 = np.eye(3)
A2[1:, 1:] = s2_fcts.fR(2 * np.pi / 6)
As = [A1, A2]

# Apply Transformations:
gmms_lt = []
for i, g in enumerate(gmms):
    A = As[i]
    b = bs[i]
    gtmp = g.copy()

    # Apply changes to the copy
    gtmp.tangent_action(A)  # Rotate Gaussian in the tangent space of the orign
    gtmp.parallel_transport(b)  # Move origin
    gmms_lt.append(gtmp)

plt.figure(figsize=(5, 5))
ax = plt.subplot(111, projection='3d')
s2_fcts.plot_manifold(ax)
for i, gmm in enumerate(gmms_lt):
    # Plot Gaussians
    for j, g in enumerate(gmm.gaussians):
        g = g.margin(1)
        s2_fcts.plot_gaussian(ax, g.mu, g.sigma, color=cols[i, :], showtangent=False)
        ax.text(g.mu[0], g.mu[1], g.mu[2], '{0}'.format(j + 1))
    ax.view_init(30, 60)
plt.show()
tin = tpdata[0][0:200]
g1 = gmms_lt[0].gmr(tin, i_in=0, i_out=1)
g2 = gmms_lt[1].gmr(tin, i_in=0, i_out=1)

g1g2 = []
for i in range(len(g1)):
    g1g2.append(g1[i] * g2[i])

plt.figure(figsize=(5,5) )
ax = plt.subplot(111,projection='3d')
s2_fcts.plot_manifold(ax)
for i, gmm in enumerate(gmms_lt):
    # Plot Gaussians
    for j,g in enumerate(gmm.gaussians):
        g = g.margin(1)
        s2_fcts.plot_gaussian(ax,g.mu,g.sigma,color=cols[i,:],showtangent=False)

# Plot the tubes of Gaussian Resulting from GMR on g1 and g2:
for i, gmm in enumerate([g1,g2]):
    for g in gmm:
        pbdplt.plot_gaussian_s2(ax,g.mu,g.sigma,color=cols[i,:],
                               showtangent=False,linealpha=0.1)
# Plot the result of the product of Gaussian:
for j,g in enumerate(g1g2):
    s2_fcts.plot_gaussian(ax,g.mu,g.sigma,color=cols[5,:],showtangent=False,
                         linealpha=0.2)
ax.view_init(30,60)
plt.show()
# plt.imshow(m_time)
# plt.colorbar()
# plt.show()
#
# plt.imshow(m_quat)
# plt.colorbar()
# plt.show()
#
# plt.imshow(m_s2)
# plt.colorbar()
# plt.show()