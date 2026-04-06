############################################
# 0 Fore-Mid brain
# 1 Forebrain o2 response
# 2 nose
# 3 ARTR R
# 4 ARTR L
# 5 Forebrain
# 6 SloMO
# 7 LMO
# 8 Midbrain
# 9 NTS

############################################
# fish 01
# cluster set 1
cell_idx = np.where(d_prime_vec<-0.8)[0]
cluster_cell_idx = []
for n in [0, 1, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])

# cluster set 2
cell_idx = np.where(d_prime_vec>0.3)[0]
for n in [0, 1, 2]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])

# cluster set 3
cell_idx = np.where(d_prime_vec<-0.5)[0]
for n in [3, 4]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])


# cluster set 4
cell_idx = np.where(d_prime_vec<-0.8)[0][ind_slim_large==0]
for n in [0, 1]:
    cluster_cell_idx.append(cell_idx[ind_slim_small==n])



############################################
# fish 00 [Drop]
# cluster set 1
d_prime_idx = d_prime_vec<-0.3
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
for n in [5, 1]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))

# cluster set 2
d_prime_idx = d_prime_vec>0.1
cell_idx = np.where(d_prime_idx)[0]
for n in [1, 0, 4]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 3
d_prime_idx = d_prime_vec<-0.3
cell_idx = np.where(d_prime_idx)[0]
for n in [3, 0, 2, 4]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])


############################################
# fish 02
# cluster set 1
d_prime_idx = d_prime_vec<-0.5
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
for n in [0, 1, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])

# cluster set 2
d_prime_idx = d_prime_vec>0.1
cell_idx = np.where(d_prime_idx)[0]
for n in [3, 2, 5]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 3
d_prime_idx = d_prime_vec<-0.4
cell_idx = np.where(d_prime_idx)[0]
for n in [0]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))

cell_idx = np.where(d_prime_vec<-0.6)[0][ind_slim_large==2]
A_center_ = A_center[d_prime_vec<-0.6][ind_slim_large==2]
cluster_cell_idx.append(cell_idx[(ind_slim_small==0) & (A_center_[:, 0]<=40)])
d_prime_idx = d_prime_vec<-0.4
cell_idx_ = np.where(d_prime_idx)[0]
cluster_cell_idx.append(np.concatenate([cell_idx[(ind_slim_small==0) & (A_center_[:, 0]>40)], cell_idx_[ind_slim==2]]))


############################################
# fish 03 [some problem]
# cluster set 1
d_prime_idx = d_prime_vec<-0.8
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
for n in [4, 1]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))
# cluster set 2
d_prime_idx = d_prime_vec>0.3
cell_idx = np.where(d_prime_idx)[0]
for n in [1, 2, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 3
d_prime_idx = d_prime_vec<-0.5
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx.append(cell_idx[ind_slim==2])
cluster_cell_idx.append(cell_idx[ind_slim==0])
cell_idx = np.where(d_prime_idx)[0]
A_center_ = A_center[d_prime_idx]
cluster_cell_idx.append(cell_idx[(ind_slim==1) & (A_center_[:, 0]<=34)])
cluster_cell_idx.append(cell_idx[(ind_slim==1) & (A_center_[:, 0]>34)])

############################################
# fish 04
# cluster set 1
d_prime_idx = d_prime_vec<-0.5
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
for n in [2, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))
# cluster set 2
d_prime_idx = d_prime_vec>0.3
cell_idx = np.where(d_prime_idx)[0]
for n in [2, 4, 6]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 3
d_prime_idx = d_prime_vec<-0.2
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx.append(cell_idx[ind_slim==3])
cluster_cell_idx.append(np.array([-1]))
cell_idx = np.where(d_prime_vec<-0.2)[0]
A_center_ = A_center[d_prime_vec<-0.2]
cluster_cell_idx.append(cell_idx[(ind_slim==3) & (A_center_[:, 0]<=40)])
cluster_cell_idx.append(cell_idx[(ind_slim==3) & (A_center_[:, 0]>40)])

############################################
# fish 05
# cluster set 1
d_prime_idx = d_prime_vec<-0.7
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
for n in [0, 1, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 2
d_prime_idx = d_prime_vec>0.4
cell_idx = np.where(d_prime_idx)[0]
for n in [2, 4, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 3
d_prime_idx = d_prime_vec<-0.3
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx[0] = cell_idx[ind_slim==0]
cluster_cell_idx.append(np.array([-1]))
cluster_cell_idx.append(cell_idx[ind_slim==3])
cluster_cell_idx.append(cell_idx[(ind_slim==1) & (A_center_[:, 0]<=34)])
cluster_cell_idx.append(cell_idx[(ind_slim==1) & (A_center_[:, 0]>34)])


############################################
# fish 06
# cluster set 1
d_prime_idx = d_prime_vec<-0.5
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
cluster_cell_idx.append(np.array([-1]))
for n in [4]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))
# cluster set 2
d_prime_idx = d_prime_vec>0.2
cell_idx = np.where(d_prime_idx)[0]
for n in [0, 1]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))
# cluster set 3
d_prime_idx = d_prime_vec<-0.2
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx.append(cell_idx[ind_slim==6])
cluster_cell_idx.append(np.array([-1]))
cell_idx = np.where(d_prime_vec<-0.2)[0]
A_center_ = A_center[d_prime_vec<-0.2]
cluster_cell_idx.append(cell_idx[(ind_slim==3) & (A_center_[:, 0]<=40)])
cluster_cell_idx.append(np.concatenate([cell_idx[(ind_slim==3) & (A_center_[:, 0]>40)], cell_idx[ind_slim==1]]))


############################################
# fish 07
# cluster set 1
d_prime_idx = d_prime_vec<-0.7
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
cluster_cell_idx.append(np.array([-1]))
for n in [2]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))
# cluster set 2
d_prime_idx = d_prime_vec>0.1
cell_idx = np.where(d_prime_idx)[0]
for n in [1, 2, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 3
d_prime_idx = d_prime_vec<-0.4
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx[2] = cell_idx[(ind_slim==5)]
cluster_cell_idx[0] = cell_idx[(ind_slim==2)]
d_prime_idx = d_prime_vec<-0.4
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx.append(cell_idx[ind_slim==4])
cluster_cell_idx.append(np.array([-1]))
cell_idx = np.where(d_prime_vec<-0.4)[0]
A_center_ = A_center[d_prime_vec<-0.4]
cluster_cell_idx.append(cell_idx[(ind_slim==1) & (A_center_[:, 0]<=40)])
cluster_cell_idx.append(np.concatenate([cell_idx[(ind_slim==1) & (A_center_[:, 0]>40)], cell_idx[ind_slim==0]]))

############################################
# fish 08
# cluster set 1
d_prime_idx = d_prime_vec<-0.7
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx = []
cluster_cell_idx.append(np.array([-1]))
for n in [0, 2]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
# cluster set 2
d_prime_idx = d_prime_vec>0.1
cell_idx = np.where(d_prime_idx)[0]
for n in [1, 3]:
    cluster_cell_idx.append(cell_idx[ind_slim==n])
cluster_cell_idx.append(np.array([-1]))
# cluster set 3
d_prime_idx = d_prime_vec<-0.4
cell_idx = np.where(d_prime_idx)[0]
cluster_cell_idx.append(cell_idx[ind_slim==1])
cluster_cell_idx.append(np.array([-1]))
cell_idx = np.where(d_prime_vec<-0.4)[0]
A_center_ = A_center[d_prime_vec<-0.4]
cluster_cell_idx.append(cell_idx[(ind_slim==0) & (A_center_[:, 0]<=34)])
cluster_cell_idx.append(np.concatenate([cell_idx[(ind_slim==3) & (A_center_[:, 0]>34)], cell_idx[ind_slim==1]]))

