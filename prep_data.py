import trimesh, glob
import numpy as np

SIZE = 2**11
KNNS = np.zeros((SIZE,SIZE),dtype=int)
knn_K = 64

def triangle_area(v1,v2,v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis = 1)

def get_pc(mesh, random_num_pts = False, n = SIZE, use_seed = True):
    if use_seed:
        np.random.seed(2)
    faces = mesh.faces
    v1s,v2s,v3s = mesh.vertices[faces[:,0],:],mesh.vertices[faces[:,1],:],mesh.vertices[faces[:,2],:]

    areas = triangle_area(v1s,v2s,v3s)

    probs = areas/areas.sum()
    # number of points to sample
    if random_num_pts == True:
        n = np.random.randint(2**12,2**16)
    weighted_rand_inds = np.random.choice(range(len(areas)),size = n, p = probs)

    sel_v1s = v1s[weighted_rand_inds]
    sel_v2s = v2s[weighted_rand_inds]
    sel_v3s = v3s[weighted_rand_inds]

    # barycentric co-ords
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)

    invalids = u + v >1 # u+v+w =1

    u[invalids] = 1 - u[invalids]
    v[invalids] = 1 - v[invalids]
    w = 1-(u+v)

    pt_cld = (sel_v1s * u) + (sel_v2s * v) + (sel_v3s * w)   
    diffs = np.max(pt_cld,axis=0) - np.min(pt_cld,axis=0)
    span_dir = np.argmax(diffs)

    return span_dir, pt_cld

def populate_knn(pc):
    i = 0
    for pt in pc:
        diffs = pc - pt
        dists = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2 + diffs[:,2]**2)
        idxs = np.argsort(dists)
        KNNS[i,:] = idxs
        i+=1

def get_knn(pt_idx, K=knn_K):      
    return KNNS[pt_idx,1:K+1]   # 1st element will be itself
    
def get_kd_idx(pts):  
     # to keep track of which points the leaves correspond to
    inds = np.array(range(len(pts)))
    ind_nodes = [inds]   
    # the orientations also give useful information
    kd_orients = []

     # 'nodes' would be sets of points, whose size reduces as the loop runs, ultimately having leaves
    nodes = [pts]
    # this will store the 'children' for an iteration, and replace node after the iteration
    child_list = []
    ind_child_list = []

    while len(nodes[0]) != 1:
        child_list.clear()
        ind_child_list.clear()
        for node,ind_node in zip(nodes,ind_nodes):  # split each node
            ranges = np.amax(node,axis = 0) - np.amin(node,axis = 0)
            split_dir = np.argmax(ranges)
            kd_orients.append(split_dir)
            # indices to be used for sorting; will use it on ind_of_original as well
            sort_ind = node[:,split_dir].argsort()

            sorted_node = node[sort_ind]
            sorted_ind_node = ind_node[sort_ind]

            num_pts = len(sorted_node)

            first_split_end = np.int(num_pts/2)

            child_list.append(sorted_node[0:first_split_end,:])
            child_list.append(sorted_node[np.int(num_pts/2):num_pts,:])
            ind_child_list.append(sorted_ind_node[0:first_split_end])
            ind_child_list.append(sorted_ind_node[np.int(num_pts/2):num_pts])
        nodes.clear()
        nodes.extend(child_list)
        ind_nodes.clear()
        ind_nodes.extend(ind_child_list)

    kd_inds = np.array(ind_nodes).reshape((len(ind_nodes),))
    return kd_inds     

def process_model(fname):
    mesh = trimesh.load_mesh(fname)
    SPAN_DIR, pc = get_pc(mesh)
    populate_knn(pc)

    idxs = get_kd_idx(pc)   

    data = np.zeros((SIZE,knn_K,3))

    for i in range(SIZE):
        idx = idxs[i]
        pt = pc[idx,:]
        knn = get_knn(idx)
        knn_pts = pc[knn]
        data[i,:,:] = knn_pts - pt
    newName = fname[:-3] + 'npy'
    np.save(newName, data)
    return newName
   
if __name__ == '__main__':
    train_models = []
    eval_models = []

    # trainFiles = glob.glob('ModelNet10/*/train/*.off')
    # for tf in trainFiles:
    #     train_models.append(process_model(tf))
    #     print(tf)

    evalFiles = glob.glob('ModelNet10/*/test/*.off')
    for ef in evalFiles:
        eval_models.append(process_model(ef))
        print(ef)

    # train_models = np.array(train_models)
    # train_models = train_models[np.random.permutation(len(train_models))]   # shuffling
    # train_models = train_models[np.random.permutation(len(train_models))]   # shuffling
    # train_models = train_models[np.random.permutation(len(train_models))]   # shuffling
    # with open('mn10train.txt','w') as f:
    #     for tm in train_models:
    #         f.write(tm + '\n')
    with open('mn10eval.txt','w') as f:
        for em in eval_models:
            f.write(em + '\n')




