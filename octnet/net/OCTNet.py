
emb_comp=5
# W = embed_trajs(data[:5], emb_comp)
W_adjusted = embed_trajs(data, emb_comp)


# In[327]:


def weights_to_cords(w, length):
    w_x = w[ : emb_comp]
    w_y = w[emb_comp : ]
    taus = np.array([(i+1)/length for i in range(length)])
    x_predict = np.array([ w_x.T.dot(k(tau,emb_comp).T) for tau in taus]).reshape(length)
    y_predict = np.array([ w_y.T.dot(k(tau,emb_comp).T) for tau in taus]).reshape(length)
    return x_predict, y_predict
xxx = []
yyy = []
zzz = []
xxxx = []
yyyy = []
zzzz = []
for i in range(5):
    t0 = data[i].trajectories[6]
    discret_taus = np.array([(i+1)/len(t0.x) for i in range(len(t0.x))])

    continue_taus = np.linspace(0., 1., 1000)
    xx, yy = weights_to_cords(W_adjusted[i][6], len(continue_taus))

    zzz.append(continue_taus)
    xxx.append(xx)
    yyy.append(yy)
    
    zzzz.append(discret_taus)
    xxxx.append(t0.x)
    yyyy.append(t0.y)
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax1.scatter(discret_taus, t0.x)
    ax1.plot(continue_taus, xx)
    plt.ylabel('x')
    plt.xlabel('t')

    ax2 = fig.add_subplot(122)
    ax2.scatter(discret_taus, t0.y)
    ax2.plot(continue_taus, yy)
    plt.ylabel('y')
    plt.xlabel('t')
fig = plt.figure(figsize=(10,5))    
ax3 = fig.add_subplot(121, projection='3d')
ax3.plot_surface(np.array(xxx), np.array(yyy), np.array(zzz), cmap='inferno')


# In[317]:


W_adjusted_exp = np.array([w if w.shape[0] == 300 else np.array(np.append(w, w[:100, :], axis=0)) for w in W_adjusted])
max_w = np.max(W_adjusted_exp)

    
W_adjusted_exp_new = W_adjusted_exp/max_w
print(W_adjusted_exp_new.shape)
# W_adjusted_exp_new = W_adjusted_exp.reshape(400, -1)


# In[349]:


max_w


# In[350]:


show_random_maps_and_trajectories(data, 9, 200)


# In[319]:


show_embeded_trajs(data, W_adjusted_exp_new*max_w, 5)


# In[431]:


X_train, X_test, W_train, W_test =  split_train_test(X_stand, W_adjusted_exp_new)
print('X shape is: {}'.format(X.shape))
print('W shape is: {}'.format(W.shape))
print('W[0] shape is: {}'.format(W[0].shape))
print('X_train shape is: {}'.format(X_train.shape))
print('X_test shape is: {}'.format(X_test.shape))
print('W_train shape is: {}'.format(W_train.shape))
print('W_test shape is: {}'.format(W_test.shape))
print('mean:{}, var:{}, std:{}'.format(np.mean(W_adjusted_exp_new), np.var(W_adjusted_exp_new), np.std(W_adjusted_exp_new)))
print(W_train[0])


# In[434]:


def slice_parameter_vectors_exp(parameter_vector, comp):
    """ Returns an unpacked list of paramter vectors.
    """
    return [
        parameter_vector[:,: ALPHAS], 
        parameter_vector[:, ALPHAS : 2*comp*ALPHAS+ALPHAS], 
        parameter_vector[:,2*comp*ALPHAS+ALPHAS:]
    ]

def gnll_loss_exp(comp=COMPONENTS):
    def gnll_loss_exp_int(y, parameter_vector):
#         print(parameter_vector)
        alpha, mu, sigma = slice_parameter_vectors_exp(parameter_vector, comp) # Unpack parameter vectors
#         print('shapes a:{}, mu:{}, s:{}'.format(alpha.shape, mu.shape, sigma.shape))
        probs = p_w_tet_exp(y, sigma, mu, comp)
        loss = -likelihood(probs, alpha, comp)
        return loss
    return gnll_loss_exp_int

def likelihood(probs, alpha, comp):
    print('probs is: {}({})\n\n'.format(probs[0, 0, :], probs.shape))
#     print('probs is: {}\n\n'.format(probs.shape))
    
    probs_by_comp = tf.math.reduce_prod(probs, axis=2)
#     print('probs_by_comp is: {}({})\n\n'.format(probs_by_comp, probs_by_comp.shape))
#     print('probs_by_comp is: {}\n\n'.format(probs_by_comp.shape))
    
    alpha_t = tf.tile(alpha, [1, 300])
#     print('alpha: {}({})\n\n'.format(alpha.shape, alpha.dtype))
    
    alpha_multiply_probs_by_comp = tf.multiply(alpha_t, probs_by_comp)
#     print('alpha_multiply_probs_by_comp is: {}({})\n\n'.format(alpha_multiply_probs_by_comp, alpha_multiply_probs_by_comp.shape))
#     print('alpha_multiply_probs_by_comp is: {}\n\n'.format(alpha_multiply_probs_by_comp.shape))
    alpha_multiply_probs_by_comp_resh = tf.reshape(alpha_multiply_probs_by_comp, [-1, 300, ALPHAS])
#     print('alpha_multiply_probs_by_comp_resh is: {}\n\n'.format(alpha_multiply_probs_by_comp_resh.shape))
    
    res = tf.math.reduce_sum(alpha_multiply_probs_by_comp_resh, axis=2)
#     print('res is: {}({})\n\n'.format(res, res.shape))
#     print('res is: {}\n\n'.format(res.shape))
    
    log_res = tf.math.log(res)
#     print('log_res is: {}({})\n\n'.format(log_res, log_res.shape))
#     print('log_res is: {}\n\n'.format(log_res.shape))
    
    aggregated_res = tf.reduce_mean(log_res, axis=1)
#     print('aggregated_res is: {}({})\n\n'.format(aggregated_res, aggregated_res.shape))
#     print('aggregated_res is: {}\n\n'.format(aggregated_res.shape))
    tf.debugging.check_numerics(aggregated_res, 'not only numbers')
    return aggregated_res
    
def p_w_tet_exp(y, sigma, mu, comp):
#     print('y_ini: {}({})({})\n\n'.format(y, y.shape, y.dtype))
#     print('alpha: {}({})({})\n\n'.format(tf.constant(alpha[0,:]), alpha.shape, alpha.dtype))
#     print('mu: {}({})\n\n'.format(mu.shape, mu))
#     print('sigma: {}({})\n\n'.format(sigma.shape, sigma))
    
    y_cast = tf.cast(y, dtype=tf.float32)
#     print('y_casted: {}({})({})\n\n'.format(y_cast, y_cast.shape, y_cast.dtype))
    y_rep = tf.repeat(y_cast, ALPHAS, axis=1)
    y_resh = tf.reshape(y_rep, [-1, ALPHAS*300, 2*comp])
#     print('y_resh: {}({})({})\n\n'.format(y_resh, y_resh.shape, y_resh.dtype))
    sigma_r = tf.reshape(sigma, [-1, ALPHAS, 2*comp])
    sigma_t = tf.tile(sigma_r, [1, 300, 1])
    
    mu_r = tf.reshape(mu, [-1, ALPHAS, 2*comp])
    mu_t = tf.tile(mu_r, [1, 300, 1])
#     print('mu_t: {}({})\n\n'.format(mu_t, mu.shape))
#     print('sigma: {}({})\n\n'.format(sigma.shape, sigma.dtype))
    double_sigma = 2*sigma_t
#     print('double_sigma is: {}({})\n\n'.format(double_sigma, double_sigma.shape))
#     print('double_sigma is: {}\n\n'.format(double_sigma.shape))
    
    y_minus_mu = y_resh - mu_t
#     print('y_minus_mu is: {}({})\n\n'.format(y_minus_mu, y_minus_mu.shape))
#     print('y_minus_mu is: {}\n\n'.format(y_minus_mu.shape))
    
    abs_y_minus_mu = tf.math.abs(y_minus_mu)
#     print('abs_y_minus_mu is: {}({})\n\n'.format(abs_y_minus_mu, abs_y_minus_mu.shape))
#     print('abs_y_minus_mu is: {}\n\n'.format(abs_y_minus_mu.shape))
    
    abs_divide_sigma =  abs_y_minus_mu / sigma_t
#     print('abs_divide_sigma is: {}({})\n\n'.format(abs_divide_sigma, abs_divide_sigma.shape))
#     print('abs_divide_sigma is: {}\n\n'.format(abs_divide_sigma.shape))
    
    negative_division = -abs_divide_sigma
#     print('negative_division is: {}({})\n\n'.format(negative_division[0, 0, :], negative_division.shape))
#     print('negative_division is: {}\n\n'.format(negative_division.shape))
    
    exponent = tf.math.exp(negative_division)
#     print('exponent is: {}({})\n\n'.format(exponent, exponent.shape))
#     print('exponent is: {}\n\n'.format(exponent.shape))
    
    probs =  exponent / double_sigma 
    return probs

model = build_static_model(5)     
model.summary()
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
)

opt = tf.optimizers.Adam(1e-3)
loss_func = gnll_loss_exp(5)
model.compile(loss=loss_func, optimizer=opt)

x = tf.constant(X_stand[:32])
w = tf.constant(W_adjusted_exp_new[:32])
# with tf.GradientTape() as tape:
#   # Forward pass
#   y = model(x)
#   loss = loss_func(w, y)

# # Calculate gradients with respect to every trainable variable
# grad = tape.gradient(loss, x)
# print(grad)

hist = model.fit(x=X_train, y=W_train, validation_data=(X_test, W_test), epochs=10, verbose=2, batch_size=32)


# In[428]:


model_exp = build_static_model(50)   

optimizer = tf.keras.optimizers.Adam()
loss_func = gnll_loss_exp(50)

loss_history = []

def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model_exp(x, training=True)
#         print(logits)
        loss = loss_func(y, logits)

    loss_history.append(loss.numpy().mean())
    grads = tape.gradient(loss, model_exp.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_exp.trainable_variables))
batch = 32    
def train(epochs):
    for epoch in range(epochs):
        train_step(X_stand[epoch*batch:(epoch+1)*batch], W_adjusted_exp_new[epoch*batch:(epoch+1)*batch])
        print ('Epoch {} finished'.format(epoch))
        
train(10)
plt.plot(loss_history)


# In[437]:


hist.__dict__
plt.plot(hist.epoch, np.array(hist.history['loss'])+5)
plt.plot(hist.epoch, np.array(hist.history['val_loss'])+5)


# In[461]:


from scipy.stats import laplace
start_time = time.time()
for entry in X_stand[:100]:
    input1 = np.array([entry])
    prediction = model.predict(input1[:1])
    alph, mu,  sig = slice_parameter_vectors_exp(prediction, 5)
    alph = alph.reshape(-1)
    mu = mu.reshape(-1, 10)
    sig = sig.reshape(-1, 10)
    t = np.arange(0, 10)


    w_pred_xs = []
    w_pred_ys = []  
    for i in range(100):
        randNum = randNum=np.random.uniform(low=0.0, high=1.0)
        best = 0
        cumulativeVal = 0
        cumulativeValUpper = 0
        for i in range(len(alph)):
            cumulativeValUpper+=alph[i]
            if((randNum>cumulativeVal)and (randNum<=cumulativeValUpper)):
                best=i
            cumulativeVal+=alph[i]  

        changed_mu = mu[best]
        changed_sig = sig[best]

        changed_mu_x = changed_mu[:5]
        changed_mu_y = changed_mu[5:]

        changed_sig_x = changed_sig[:5]
        changed_sig_y = changed_sig[5:]

        w_pred_x = []
        w_pred_y = []

        for i, m in enumerate(changed_mu_x):
            res_x = (m + laplace.rvs(loc=0, scale=changed_sig_x[i]))*max_w
            w_pred_x.append(res_x) 

        for i, m in enumerate(changed_mu_y):
            res_y = (m + laplace.rvs(loc=0, scale=changed_sig_y[i]))*max_w
            w_pred_y.append(res_y) 

        w_pred_xs.append(w_pred_x)
        w_pred_ys.append(w_pred_y)


    w_pred_xs_np = np.array(w_pred_xs)
    w_pred_ys_np = np.array(w_pred_ys)

# plt.imshow(np.abs(np.array(data[0].map, dtype='float32') - 1), cmap='gray')
# for i in range(len(w_pred_xs_np)):  
#     w_x = w_pred_xs_np[i]
#     w_y = w_pred_ys_np[i]
    
#     taus = np.array([(i+1)/len(t) for i in range(len(t))])
    
#     x_predict = np.array([ w_x.T.dot(k(tau,5).T) for tau in taus]).reshape(len(t))
#     y_predict = np.array([ w_y.T.dot(k(tau,5).T) for tau in taus]).reshape(len(t))
#     if np.min(x_predict)>0 and np.max(x_predict)<40 and np.min(y_predict)>0 and np.max(y_predict)<40:
#         plt.plot(x_predict, y_predict)

print('{} s)'.format(time.time() - start_time))


# In[392]:


plt.figure(figsize=(10,10))
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(np.abs(np.array(data[100].map, dtype='float32') - 1), cmap='gray')
randT = np.random.randint(0, 200, 200)
traj_to_print = []
for r in randT:
    trj = data[100].trajectories[r]
    traj_to_print.append(trj)
    
for t in traj_to_print:
    plt.plot((t.x+laplace.rvs(loc=0, scale=changed_sig_y[0])*3), (t.y+laplace.rvs(loc=0, scale=changed_sig_y[0])*2))
for t in traj_to_print:
    plt.plot((t.x+laplace.rvs(loc=0, scale=changed_sig_y[0])*2), (t.y+laplace.rvs(loc=0, scale=changed_sig_y[0])*3))


# In[393]:


import random
import time

def filter_to_shapes(filt):
    vert = []
    for i_r, r in enumerate(filt):
        if len(np.unique(r)) > 1:
            vert.append(i_r) #add top and bottom
            
    filt_t = filt.T
    for i_r, r in enumerate(filt_t):
        if len(np.unique(r)) > 1:
            vert.append(i_r) #add left and right
    #vert is [top, bottom, left, right]   
    vert[0] = vert[0] - 1
    vert[1] = vert[1] + 1
    vert[2] = vert[2] - 1
    vert[3] = vert[3] + 1
    return np.array([
            [
                [vert[2], vert[1]],
                [vert[2], vert[0]]
            ], #left
            [
                [vert[2], vert[0]],
                [vert[3], vert[0]]
            ], #top
            [
                [vert[3], vert[0]], 
                [vert[3], vert[1]]
            ], #right
            [
                [vert[3], vert[1]], 
                [vert[2], vert[1]]
            ] #bottom
        ])
    
def is_dot_in_shape(x, y, shape):
    mp1 = [(shape[0][0] - x), (shape[0][1] - y)]
    mp2 = [(shape[1][0] - x), (shape[1][1] - y)]
    return np.dot(mp1, mp2) <= 0
    
    
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return is_dot_in_shape(x,y,line1) and is_dot_in_shape(x,y,line2)            
    
            
    
def trajectory_to_shapes(trajectory):
    x = trajectory.x
    y = trajectory.y
    lines = []
    for i in range(len(x) - 1):
        l_x_1 = x[i]
        l_x_2 = x[i+1]
        l_y_1 = y[i]
        l_y_2 = y[i+1]
        lines.append([[l_x_1, l_y_1], [l_x_2, l_y_2]])
    return np.array(lines)
    
def is_traj_intersect_filter(traj, filt):
    result = False
    for filt_shape in filt:
        for traj_shape in traj:
            inter = line_intersection(filt_shape, traj_shape)
            result = result or inter
    return result
    
def wrapped_init(d, num, map_ini, filt, lines, colors):
    def init():
        filt = dot_filter()
        map_mod = apply_filter(map_ini, filt)
        map_to_show = np.abs(map_mod - 1)

        traj_shapes = []
        for t in d.trajectories:
            traj_shapes.append(trajectory_to_shapes(t))
        filter_shapes = filter_to_shapes(filt)    
        to_draw = []
        for i_t, t in enumerate(traj_shapes):
            if not is_traj_intersect_filter(t, filter_shapes):
                to_draw.append(d.trajectories[i_t])

        for l in lines:
            l[0].set_data([], [])
        im.set_data(map_to_show)
        for i_t, t in enumerate(to_draw):  
            lines[i_t][0].set_data(t.x, t.y)
            lines[i_t][0].set_color(colors[i_t])

        return [im].append(lines)
    return init
    
def wrapped_animate(d, num, map_ini, filt, lines, colors):
    def animate(i):
        filt = move_filter(dot_filter(), i)
        map_mod = apply_filter(map_ini, filt)
        map_to_show = np.abs(map_mod - 1)

        traj_shapes = []
        for t in d.trajectories:
            traj_shapes.append(trajectory_to_shapes(t))
        filter_shapes = filter_to_shapes(filt)    
        to_draw = []
        for i_t, t in enumerate(traj_shapes):
            if not is_traj_intersect_filter(t, filter_shapes):
                to_draw.append(d.trajectories[i_t])

        for l in lines:
            l[0].set_data([], [])
        im.set_data(map_to_show)
        for i_t, t in enumerate(to_draw):  
            lines[i_t][0].set_data(t.x, t.y)
            lines[i_t][0].set_color(colors[i_t])

        return [im].append(lines)
    return animate

def animate_maps(amount):
    if amount > len(data):
        amount = len(data)
        
    for num in range(amount):
        d = data[num]
        map_ini = np.array(d.map, dtype='int32')
        map_mod = apply_filter(map_ini, dot_filter())
        map_to_show = np.abs(map_mod - 1)

        fig = plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        im = plt.imshow(map_to_show, cmap='gray')  
        lines = [plt.plot([], [], alpha=1, animated=True) for i in range(len(d.trajectories))]
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(d.trajectories))]

        init_fun = wrapped_init(d, num, map_ini, filt, lines, colors)
        animate_fun = wrapped_animate(d, num, map_ini, filt, lines, colors)

        anim = FuncAnimation(fig, animate_fun, init_func=init_fun, frames=20, interval=500, blit=False)
        anim.save('map_{}_anim.gif'.format(num), writer='imagemagick')  
    

def generate_maps(data, amount): #generate amount+amount*20 maps and trajectories
    generated = []
    if amount > len(data):
        amount = len(data)
        
    for num in range(amount):
        start_time = time.time()
        d = data[num]
        map_ini = np.array(d.map, dtype='int32')
        traj_ini = d.trajectories
            
        generated.append(DataUnit(Map(map_ini.tolist()), traj_ini))
            
        for step in range(20):
            filt = move_filter(dot_filter(), step)
            map_mod = apply_filter(map_ini, filt)

            traj_shapes = []
            for t in traj_ini:
                traj_shapes.append(trajectory_to_shapes(t))
                    
            filter_shapes = filter_to_shapes(filt) 
                
            traj_mod = []
            for i_t, t in enumerate(traj_shapes):
                if not is_traj_intersect_filter(t, filter_shapes):
                    traj_mod.append(traj_ini[i_t])
            
            generated.append(DataUnit(Map(map_mod.tolist()), traj_mod))
        print('{}/{} ({} s)'.format((num+1), amount, (time.time() - start_time)))
    return generated

generated_data = generate_maps(data, 1)
# generated_data2 = generate_maps(data2, 400)


# In[ ]:





# In[450]:


fig = plt.figure(figsize=(20,5))
for i, m in enumerate(generated_data[1:5]):
    map_i = m.map
    fig.add_subplot(1,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.abs(np.array(map_i, dtype=np.float32) - 1), cmap='gray')
    for t in m.trajectories[:20]:
        plt.plot((t.x+changed_sig_y[0]*9), (t.y+changed_sig_y[0]*8))


# In[414]:


fig = plt.figure(figsize=(10,10))
plt.xticks([])
plt.yticks([])
plt.grid(False)
map_ini = generated_data[1].map
trah_ini = generated_data[1].trajectories
filt = dot_filter()

map_to_show = apply_filter(map_ini, filt)

shape_f = filter_to_shapes(filt)
shape_t = [trajectory_to_shapes(t) for t in trah_ini]

plt.imshow(np.abs(np.array(map_to_show, dtype=np.float32) - 1), cmap='gray')

for pairs in shape_f:
    plt.plot([pairs[0][0], pairs[1][0]], [pairs[0][1], pairs[1][1]])
    
    
for v in shape_t:
    for pairs in v:
        plt.plot([pairs[0][0], pairs[1][0]], [pairs[0][1], pairs[1][1]])


# In[185]:


show_random_maps_and_trajectories(generated_data, 9, 100)


# In[196]:


traj_dist = []
for d in data:
    for t in d.trajectories:
        traj_dist.append(len(t.x))
figure = plt.figure(figsize=(10, 10))
plt.hist(np.array(traj_dist))
plt.ylabel('Количество траекторий')
plt.xlabel('Длина')
# plt.legend()


# In[215]:


fig = plt.figure(figsize=(20, 5))
fig.add_subplot(1,4,1)    
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(np.abs(np.array(data[0].map, dtype='float32') - 1), cmap='gray')
for t in data[0].trajectories:
    plt.plot(t.x, t.y)
       
fig.add_subplot(1,4,2)    
d2 = np.rot90(data[0].map, k=1, axes=(0, 1))
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(np.abs(np.array(d2, dtype='float32') - 1), cmap='gray')
for t in data[0].trajectories:
    t_x = 40 - np.array(t.x)
    t_y = np.array(t.y)
    plt.plot(t_y, t_x)
            
fig.add_subplot(1,4,3)    
d3 = np.rot90(data[0].map, k=2, axes=(0, 1))
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(np.abs(np.array(d3, dtype='float32') - 1), cmap='gray')
for t in data[0].trajectories:
    t_x = 40 - np.array(t.x)
    t_y = 40 - np.array(t.y)
    plt.plot(t_x, t_y)
    
fig.add_subplot(1,4,4)    
d4 = np.rot90(data[0].map, k=3, axes=(0, 1))
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(np.abs(np.array(d4, dtype='float32') - 1), cmap='gray')
for t in data[0].trajectories:
    t_x = np.array(t.x)
    t_y = 40 - np.array(t.y)
    plt.plot(t_y, t_x)


# In[268]:


# fig = plt.figure(figsize=(10,10))
# for d in data[0].trajectories:
#     ind = [i for i in range(len(d.x))]
#     plt.scatter(ind, d.x)
    
fig = plt.figure(figsize=(10,10))
x_s = [d.x for d in data[0].trajectories]
plt.hist(x_s)

x = np.linspace(0, 35, 350)
plt.plot(x, laplace.pdf(x, loc=3.5, scale=0.5)*4, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=6, scale=0.5)*4, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=9, scale=0.5)*7, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=12, scale=0.5)*6, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=15, scale=0.5)*7, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=18, scale=0.5)*8, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=21, scale=0.5)*12, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=24, scale=0.5)*7, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=27, scale=0.5)*7, color='green', lw=3)

plt.plot(x, laplace.pdf(x, loc=30, scale=0.5)*10, color='green', lw=3)

plt.ylabel('Число совпадений')
plt.xlabel('Координата x')


# In[275]:


fig = plt.figure(figsize=(10,10))
maps = [np.array(d.map).flatten() for d in data]
plt.hist(maps)
plt.ylabel('Количество')
plt.xlabel('Значение по картам')


# In[276]:


fig = plt.figure(figsize=(10,10))
plt.hist(X)
plt.ylabel('Количество')
plt.xlabel('Значение по картам')


# In[289]:


dddd = data[0].trajectories

tsx = [t.x for t in dddd]
tsy = [t.y for t in dddd]
fig = plt.figure(figsize=(10,5))
fig.add_subplot(121)
plt.hist(tsx)
plt.ylabel('Количество')
plt.xlabel('x')
fig.add_subplot(122)
plt.hist(tsy)
plt.ylabel('Количество')
plt.xlabel('y')


# In[290]:


wsx = W_adjusted_exp_new[0][:50]
wsy = W_adjusted_exp_new[0][50:]
fig = plt.figure(figsize=(10,5))
fig.add_subplot(121)
plt.hist(wsx)
plt.ylabel('Количество')
plt.xlabel('w(x)')
fig.add_subplot(122)
plt.hist(wsy)
plt.ylabel('Количество')
plt.xlabel('w(y)')

