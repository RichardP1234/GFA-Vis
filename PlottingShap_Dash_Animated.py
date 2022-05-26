"""
Creating interactive figure to show most important features

@author: Richard Pyle
"""
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly import subplots

#%% Load in an image, feature and Shapley values
indexes = np.load('Data/indexes_XTest_E.npy')
index = np.random.randint(len(indexes))

image = np.load('Data/images.npy')[index]
image_W = np.load('Data/images_W.npy')[index]
#image_Gauss = np.load('recons.npy')[index]
feat = np.load('Data/Feat.npy')[index]
Y = np.load('Data/Y.npy')[index]
ii = np.load('Data/indexes_XTest_E.npy')[index]
shap_values = np.abs(np.load('Data/shap_values.npy')[index])
mid = np.load('Data/mids.npy')[index]
pred = np.load('Data/Preds.npy')[index]
ref = np.load('Data/max_PWI_S.npy')


height = int(20) #pixels
width = int(10)
#%% Functions

def twoD_Gaussian(xy, *args): #amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    x,y = xy
    N_GAUSS = int((len(args)-7)/6)+1
    g = np.zeros(x.shape)
    for G in range(N_GAUSS):
        if G == 0:
            amplitude, xo, yo, sigma_x, sigma_y, theta, offset = args[:7]
        else:
            amplitude, xo, yo, sigma_x, sigma_y, theta = args[7+(G-1)*6:7+G*6]
            offset = 0
        #amplitude, xo, yo, sigma_x, sigma_y, theta  = args[G*6:(G+1)*6]
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g += amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2))) + offset
    return g.ravel()

def plot_wireframe(xx, yy, z, color='#0066FF', linewidth=1):
    line_marker = dict(color=color, width=linewidth)
    lines = []
    for i, j, k in zip(xx, yy, z):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker,showlegend=False,hoverinfo='skip'))
    for i, j, k in zip(np.swapaxes(xx,0,1), np.swapaxes(yy,0,1), np.swapaxes(z,0,1)):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker,showlegend=False,hoverinfo='skip'))
    return lines

def Arrow3D(xx, yy, zz, color='#0066FF', linewidth=1,arrow_tip_size = 1,reverse=False,max_tip_perc=0.3,end=True,begin=False):
    arrow = []
    if reverse:
        xx = np.flip(xx)
        yy = np.flip(yy)
        zz = np.flip(zz)
    arrow.append( go.Scatter3d(
        x=xx,
        y=yy,
        z=zz,
        mode='lines',
        line = dict(width = linewidth, color = color),
        showlegend=False,hoverinfo='none',
    ) )
    vec = np.array([(xx[1] - xx[0]),(yy[1] - yy[0]),(zz[1] - zz[0])]) 
    if arrow_tip_size > np.linalg.norm(vec)*max_tip_perc:
        arrow_tip_size = np.linalg.norm(vec)*max_tip_perc
    vec = vec / np.linalg.norm(vec)
    if end:
        head = go.Cone(anchor='tip',
            x=[xx[1]],
            y=[yy[1]],
            z=[zz[1]],
            u=[arrow_tip_size*vec[0]*2],
            v=[arrow_tip_size*vec[1]*2],
            w=[arrow_tip_size*vec[2]*2],
            showlegend=False,
            showscale=False,
            colorscale=[[0, color], [1, color]],hoverinfo='none'
            )
        arrow.append(head)
    if begin:
        foot = go.Cone(anchor='tip',
            x=[xx[0]],
            y=[yy[0]],
            z=[zz[0]],
            u=[-arrow_tip_size*vec[0]*2],
            v=[-arrow_tip_size*vec[1]*2],
            w=[-arrow_tip_size*vec[2]*2],
            showlegend=False,
            showscale=False,
            colorscale=[[0, color], [1, color]],hoverinfo='none'
            )
        arrow.append(foot)
    return arrow

def Explanation(im_orig1,im1,Feat1,mid,lw=3):

    upsample = 2
    
    ar_size_def = 1
    
    Shaps = {}
    gauss_surf = []
    flat_orig = []
    for mo in range(4):
        im_orig = np.copy(im_orig1[:,:,mo])
        Feat = np.copy(Feat1[:,mo])
        
        Feat[1] = Feat[1] - mid+width/2
        Feat[2] = 32-(height-Feat[2])
        
        im_orig_hi = np.kron(im_orig, np.ones((upsample,upsample)))
        z_hi = 32 - (height-(np.arange(0,height+1/upsample,1/upsample)))
        x_hi = np.arange(0,width+1/upsample,1/upsample)
        X_hi,Z_hi = np.meshgrid(x_hi,z_hi)
        im_recon_hi = np.reshape(twoD_Gaussian([X_hi,Z_hi], *Feat),Z_hi.shape)
        
        #Surface
        gauss_surf.append(plot_wireframe(X_hi, Z_hi, im_recon_hi, color='black', linewidth=1))
        x = np.arange(width)
        z = np.flip(32-np.arange(height))
        X,Z = np.meshgrid(x,z)

        im2 = 20*np.log10(im_orig_hi/ref)
        #Image
        flat_orig.append( go.Surface(z=np.zeros(im_orig_hi.shape),
                x=x_hi,
                y=z_hi,
                surfacecolor=im2,
                colorscale=px.colors.sequential.Jet,
                cmin = -60,cmax=0,showscale = False,opacity=0.6,hoverinfo='none'))
        

        
        amp = Feat[0]
        x0 = Feat[1]
        y0 = Feat[2]
        sigx = Feat[3]
        sigy = Feat[4]
        theta = Feat[5]
        offset = Feat[6]

        z_sig1 = np.exp(-0.5)*amp

        origin = [x_hi[0]-0.05*width,z_hi[0]-0.02*height,0]
        
        ## Create Arrows
        #x0
        xx = [origin[0],x0]
        yy = [origin[1],origin[1]]
        zz = [origin[2],origin[2]]
        Shaps['x' + str(mo+1)] = Arrow3D(xx, yy, zz,'black',lw,arrow_tip_size=ar_size_def)

        #y0
        xx = [origin[0],origin[0]]
        yy = [origin[1],y0]
        zz = [origin[2],origin[2]]
        Shaps['y' + str(mo+1)] = Arrow3D(xx, yy, zz,'black',lw,arrow_tip_size=ar_size_def)
        
        #Amp
        xx = [x0,x0]
        yy = [y0,y0]
        zz = [offset,offset+amp]
        Shaps['amp' + str(mo+1)] = Arrow3D(xx, yy, zz,'black',lw,arrow_tip_size=ar_size_def*1.5*np.max(Feat1[0,:])/width)
        
        #Sigy
        x1 = x0 - sigy*np.tan(theta)
        x2 = x0 + sigy*np.tan(theta)
        y1 = y0 - sigy*np.cos(theta)
        y2 = y0 + sigy*np.cos(theta)

        xx = [x1,x2]
        yy = [y1,y2]
        zz = [z_sig1+offset,z_sig1+offset]
        Shaps['sigy' + str(mo+1)] = Arrow3D(xx, yy, zz,'black',lw,arrow_tip_size=ar_size_def,begin=True,max_tip_perc=0.4)
        
        
        #Sigx
        x1 = x0 - sigx*np.cos(theta)
        x2 = x0 + sigx*np.cos(theta)
        y1 = y0 + sigx*np.tan(theta)
        y2 = y0 - sigx*np.tan(theta)
        
        xx = [x1,x2]
        yy = [y1,y2]
        zz = [z_sig1+offset,z_sig1+offset]
        Shaps['sigx' + str(mo+1)] = Arrow3D(xx, yy, zz,'black',lw,arrow_tip_size=ar_size_def,begin=True,max_tip_perc=0.4)
        
        
        #Theta
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return(x, y)
        theta2 = -theta-np.pi/2
        rho = 2
        of = [-1.75,1]
        xline,yline = pol2cart(rho, np.arange(0,theta2,theta2/50))
        yline += 32-of[1]
        zline = offset+np.zeros(xline.shape)
        xline -= max(xline) - of[0]
        ex = 0.4
        centre = [of[0]-rho,of[0]+ex]

        line1 = go.Scatter3d(
            x=xline,
            y=yline,
            z=zline,
            mode='lines',
            line = dict(width = lw, color = 'black'),
            showlegend=False,hoverinfo='none')
        ar1 = Arrow3D([xline[-20],xline[-1]], [yline[-20],yline[-1]], zline[-2:],'black',lw,arrow_tip_size=ar_size_def,max_tip_perc=1)
        line2 = go.Scatter3d(
            x=[centre[0],centre[1]],
            y=[np.max(yline),np.max(yline)],
            z=[zline[0],zline[0]],
            mode='lines',
            line = dict(width = lw, color = 'black'),
            showlegend=False,hoverinfo='none')
        line3 = go.Scatter3d(
            x=[centre[0],centre[1]-rho-ex+pol2cart(rho+ex, theta2)[0]],
            y=[np.max(yline),np.max(yline)+pol2cart(rho+ex, theta2)[1]],
            z=[zline[0],zline[0]],
            mode='lines',
            line = dict(width = lw, color = 'red',dash='longdash'),
            showlegend=False,hoverinfo='none')
        

        Shaps['theta' + str(mo+1)] = [line1,line2,line3,ar1[0],ar1[1]]
        
        #Offset             
        xx = [width+1.5,width+1.5]
        yy = [np.max(Z_hi)-1,np.max(Z_hi)-1]
        if offset > np.max(Feat1[0,:])*0.05:
            zz = [0,offset]
        else:
            zz = [0,offset+np.max(image_W)*0.05]
        Shaps['offset' + str(mo+1)] = Arrow3D(xx, yy, zz,'green',lw,arrow_tip_size=ar_size_def,max_tip_perc=0.7)
    
        
    return Shaps,flat_orig,gauss_surf


#%% Shapley values - > which features
fs_fancy = ['Amp', 'Pos<sub>x</sub>', 'Pos<sub>y</sub>', 'Sigma<sub>y</sub>', 'Sigma<sub>x</sub>', 'Theta', 'Offset']
feature_names_fancy = np.array([[x+' (1)' for x in fs_fancy],[x+' (2)' for x in fs_fancy],[x+' (3)' for x in fs_fancy],[x+' (4)' for x in fs_fancy]]).swapaxes(0,1)
#feature_names_fancy = list(np.reshape(feature_names_fancy,[-1]))

fs = ['amp', 'x', 'y', 'sigx', 'sigy', 'theta', 'offset']
feature_names = np.array([[x + '1' for x in fs],[x + '2' for x in fs],[x + '3' for x in fs],[x + '4' for x in fs]]).swapaxes(0,1)
feature_names = list(np.reshape(feature_names,[-1]))

fs_sort = [x for _, x in sorted(zip(np.reshape(np.abs(shap_values),[-1]), np.reshape(feature_names,[-1])))]
shap_sort = [x for _, x in sorted(zip(np.reshape(np.abs(shap_values),[-1]), np.reshape(shap_values,[-1])))]
pairs = {fs_sort[i]: shap_sort[i] for i in range(len(fs_sort))}

#Create Components of Figures
Shaps,flat_orig,gauss_surf = Explanation(image_W,image,feat,mid,lw=5)



PercAll = np.flip(np.arange(0,101,5))


specs = [[{'type': 'surface'}, {'type': 'surface'},{'type': 'surface'}, {'type': 'surface'}, {'type': 'bar'}]]
#specs = [[{'type': 'surface'}, {'type': 'surface'}],[{'type': 'surface'}, {'type': 'surface'}]]
maxcols = 5
fig = subplots.make_subplots(rows=1, cols=maxcols,
                                 specs=specs,column_widths = (1,1,1,1,0.3),
                                 subplot_titles=('$(1)\ SS-S,\ \Psi = -45^\circ$','$(2)\ SS-L,\ \Psi = -45^\circ$','$(3)\ SS-S,\ \Psi = 45^\circ$','$(4)\ SS-L,\ \Psi = 45^\circ$',''))



for mo in range(4):
    row,col = np.unravel_index(mo,[1,maxcols])
    row += 1
    col += 1
    for f in ['amp','x','y','sigy','sigx','theta','offset']:
        [fig.add_trace(l,row=row,col=col) for l in Shaps[f + str(mo+1)]]

n_traces_dims = len(fig.data)

max1 = np.max(shap_values)

colours = ['black','black','red','red']
patterns = ['','x','','x']
for mo in range(4):
    for f in range(7):
        bar = go.Bar(
                    x=[shap_values[0,f,mo]],
                    hoverlabel=dict(namelength=0),
                    y=[feature_names_fancy[f,mo]],
                    orientation='h',
                    marker=dict(color = colours[mo],
                                pattern = dict(shape = patterns[mo],solidity=.7)),
                    showlegend=False)

        print([feature_names_fancy[f,mo]])                   
        fig.add_trace(bar,row=1,col=maxcols)
n_traces_bar = len(fig.data)


for mo in range(4):
    row,col = np.unravel_index(mo,[1,maxcols])
    row += 1
    col += 1
    plotting = [flat_orig[mo]]
    [plotting.append(l) for l in gauss_surf[mo]]
    [fig.add_trace(l,row=row,col=col) for l in plotting]

n_traces = len(fig.data)
perc_shap = []
steps = []
for pp,P in enumerate((PercAll)):
    count = 0
    f_plot = []
    s_plot = []
    for ii in reversed(range(len(fs_sort))):
        if count < P+1:
            f_plot.append(fs_sort[ii])
            s_plot.append(shap_sort[ii])
        count += np.abs(shap_sort[ii]) / np.sum(np.abs(shap_values)) * 100
    perc_shap.append( np.sum(np.abs(s_plot)) / np.sum(np.abs(shap_values)) * 100 )
    step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title":"{0:.3g}% of Shapley described".format(perc_shap[pp])}
                  ],
            label = "{0:.0f}%".format(P)
        )
    cc = 0
    for mo in range(4):
        for i,f in enumerate(['amp','x','y','sigy','sigx','theta','offset']):
            add = len(Shaps[f + str(mo+1)])
            if f + str(mo+1) in f_plot:
                for a in range(add):
                    step["args"][0]["visible"][cc] = True  # Toggle cc trace to "visible"
                    cc += 1
            else:
                cc += add
    for mo in range(4):
        for i,f in enumerate(['amp','x','y','sigy','sigx','theta','offset']):
            if f + str(mo+1) in f_plot:
                step["args"][0]["visible"][cc] = True
            cc += 1
    for cc2 in range(cc,n_traces):
        step["args"][0]["visible"][cc2] = True

    steps.append(step)
    
transition = {"duration":300}    
sliders = [dict(
    active=len(steps),
    currentvalue={"visible": False},
    pad={"t": 50,},
    steps=steps,
    transition = transition,
)]
x = np.arange(width)
z = np.flip(32-np.arange(height))

fig.update_layout(
    sliders=sliders,
scene1=dict(
    xaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[x[0]-3,x[-1]+3]),
    yaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[z[0]-3,z[-1]+3]),
    zaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[0,np.max(feat[0,:]+feat[-1,:])*1.05])), #Plot Subtitle</sup>"
scene2=dict(
    xaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[x[0]-3,x[-1]+3]),
    yaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[z[0]-3,z[-1]+3]),
    zaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[0,np.max(feat[0,:]+feat[-1,:])*1.05])),
scene3=dict(
    xaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[x[0]-3,x[-1]+3]),
    yaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[z[0]-3,z[-1]+3]),
    zaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[0,np.max(feat[0,:]+feat[-1,:])*1.05])),
scene4=dict(
    xaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[x[0]-3,x[-1]+3]),
    yaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[z[0]-3,z[-1]+3]),
    zaxis=dict(showticklabels=False,visible=False,showspikes=False,range=[0,np.max(feat[0,:]+feat[-1,:])*1.05])),
height=700,width=1700)

fig['layout']['xaxis1']['title']=dict(text='Shapley Value (mm)',font=dict(size=12))
fig['layout']['xaxis1']['side'] = 'top'
fig['layout']['xaxis1']['fixedrange'] = True
fig['layout']['yaxis1']['fixedrange'] = True

zoom_in = 1.2
for mo in range(4):
    row,col = np.unravel_index(mo,[1,maxcols])
    row += 1
    col += 1
    if mo == 0 or mo == 1:
        flip=-1
    else:
        flip=-1
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=flip*0.6/zoom_in, y=1.6/zoom_in, z=1/zoom_in))
    
    
    
    fig.update_scenes(xaxis_autorange="reversed",row=row,col=col,
                        aspectratio = dict(x=len(x)/len(z), y=len(z)/len(z), z=3/ref),
                        camera=camera)








import plotly.io as pio
pio.renderers.default='browser'
fig.show()
            