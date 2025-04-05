import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 正規分布を取得する関数
def dist(x,m,s):
    d = (np.exp(-((x-m)**2)/(2*s)))/np.sqrt(2*np.pi*s)
    return d

# ガンマを計算
def calc_gamma(x,m,s,pi):
    gamma = pi*dist(x,m,s)
    gamma /= np.sum(gamma,axis=1).reshape(len(x),1)
    return gamma

# 各分布のパラメータを計算
def update(x,m,s,pi):
    gamma = calc_gamma(x,m,s,pi)
    Nk = np.sum(gamma,axis =0)
    N = np.sum(Nk)
    mu_k = np.sum(x*gamma,axis=0)/Nk
    sigma_k = np.sum(gamma*(x-mu_k)**2,axis=0)/Nk
    pi_k = Nk/N
    return mu_k,sigma_k,pi_k

# 損失関数の計算と更新
def M_step(x,m,s,pi):
    e = 0.1
    lf = 0
    for i in range(200):
        d = dist(x,m,s)
        lf_new = np.sum(np.log(np.sum(pi*d,axis=1)))
        h = lf_new-lf
        if np.abs(h)<e:
            print("iteration stop")
            break
        lf = lf_new
        m,s,pi = update(x,m,s,pi)
    return m,s,pi
    
if __name__ == '__main__':
    # loc:平均 scale:標準偏差
    x1 = np.random.normal(loc=0.5, scale=1 , size =100).reshape(-1,1)
    x2 = np.random.normal(loc=10, scale=2 , size =100).reshape(-1,1)
    x3 = np.random.normal(loc=0, scale=3 , size =100).reshape(-1,1)
    x = np.concatenate([x1 , x2 , x3])
    #sns.histplot(x, kde=True)
    #plt.title("Gaussian Mixture Model")
    #plt.show()
    m = np.array([0,10,3])
    s = np.array([1, 5, 10])
    pi = np.array([0.1,0.4, 0.5])
    m,s,pi = M_step(x,m,s,pi)
    print(m,s,pi)
    y0 = np.random.normal(loc=m[0], scale=np.sqrt(s)[0] , size =int(300*pi[0]) ).reshape(-1,1)
    y1 = np.random.normal(loc=m[1], scale=np.sqrt(s)[1] , size =int(300*pi[1]) ).reshape(-1,1)
    y2 = np.random.normal(loc=m[2], scale=np.sqrt(s)[2] , size =int(300*pi[2]) ).reshape(-1,1)
    y=np.concatenate([y0, y1, y2])
    sns.histplot(y)
    plt.title("Predicted GMM")
    plt.show()
        
    
    
    