import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import curve_fit
from functools import partial
from scipy.spatial.distance import pdist

# PR centers in sMPA coordinates
PRcenters = dict({'PR1':np.array([0,0,0.022]),
'PR2':np.array([128.367,60.092,0.015]),
'PR3':np.array([-127.980,60.029,-0.026]),
'PR4':np.array([-130.118,-66.958,0.014]),
'PR5':np.array([130.151,-67.021,-0.025])})

def calc_uvec(df,pose,length,rotangle):
    # compute GSA rotation matrix from the rx/ry values of the current pose
    rx = df.loc[pose,'Rx']
    ry = df.loc[pose,'Ry']
    gsa_rot = R.from_euler('XY',[rx+rotangle,ry], degrees=True) #Add an extra ~11 deg to account for the sMATF tilt
    uvec = np.array([0.,0.,length])
    rot_uvec = gsa_rot.apply(uvec)#np.dot(gsa_rot.as_matrix(),uvec)
    return rot_uvec

def update_uvec(df,pose,length,rotangle):
    rot_uvec = calc_uvec(df,pose,length,rotangle)
    df.loc[pose,'uvec_X'] = rot_uvec[0]
    df.loc[pose,'uvec_Y'] = rot_uvec[1]
    df.loc[pose,'uvec_Z'] = rot_uvec[2]

def update_RxRy(df,pose,GSA_angle_WCS_deg):
    # for pose in df.index if pose not in ['sMask','sMPA']:
    axis = df.loc[pose,['uvec_X','uvec_Y','uvec_Z']]
    # print(df.loc[pose,['Rx','Ry']])
    # axis=np.array([1.0,2,3])
    axis /= np.sqrt(np.sum(axis**2))
    Rx = np.arctan2(axis[1],axis[2])*180./np.pi
    # print(-Rx-GSA_angle_WCS_deg)
    # print('Original axis: ',axis)

    rot_Rx = R.from_euler('X',Rx,degrees=True)
    new_axis = rot_Rx.apply(axis.astype(float))
    # print('After Rx: ',new_axis)

    Ry = np.arctan2(new_axis[0],new_axis[2])*180./np.pi
    # print(Ry)
    rot_Ry = R.from_euler('Y',Ry,degrees=True)     #Added a negative sign here
    new_new_axis = rot_Ry.inv().apply(new_axis)
    # print('After Ry: ',new_new_axis)  #This vector should be parallel to +Z

    df.loc[pose,'Rx'] = -Rx-GSA_angle_WCS_deg
    df.loc[pose,'Ry'] = Ry
    # print(df.loc[pose,['Rx','Ry']])

def calc_RxRy_from_pose(pose_vector,gsa_offset,show_plot=False):
    #Inputs:
    #pose_vector = Array-like. Components of the pose vector. Does not have to be normalized.
    #gsa_offset = float. Value of the fixed offset in GSA encoders in units of degrees, typically ~11 degrees.

    #Returns:
    #Two element array [Rx-gsa_offset,Ry] in degrees, representing the Euler angles needed to transform the +Z axis into pose_vector.

    z_vector = np.array([0,0,1])
    pose_vector /= np.linalg.norm(pose_vector)
    #Define and apply rot_Rx
    rot_Rx = R.from_euler('X',np.arctan2(-pose_vector[1],pose_vector[2]))
    rx = rot_Rx.as_rotvec(degrees=True)[0]
    z_vector_rot = rot_Rx.apply(z_vector)
    pose_vector_updated = rot_Rx.apply(pose_vector)
    # print('Rx: ',round(rx,4),' degrees (after accounting for GSA offset)')
    print(rx)
    # print('Rx: ',rot_Rx.as_rotvec(degrees=True))
    # print(pose_vector_updated)


    if show_plot==True:
        ####Plotting####
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        # Plot the X, Y, and Z axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='+X')
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='+Y')
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='+Z')
        ax.set_xlim(-1,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)

        # Plot the pose vector and first rotated vector
        ax.plot([0, pose_vector[0]], [0, pose_vector[1]], [0, pose_vector[2]], 'ko-', label='pose vector')
        ax.plot([0, z_vector_rot[0]], [0, z_vector_rot[1]], [0, z_vector_rot[2]], 'co-', label='+Z axis (after Rx)')

    #Define and apply rot_Ry, so the z_vector is now parallel to pose_vector
    rot_Ry = R.from_euler('Y',np.arctan2(pose_vector[0],pose_vector[2]))
    ry = rot_Ry.as_rotvec(degrees=True)[1]
    z_vector_rot = rot_Ry.apply(z_vector_rot)
    print(ry)
    # print('Ry: ',round(ry,4),' degrees')

    if show_plot == True:
        ax.plot([0, z_vector_rot[0]], [0, z_vector_rot[1]], [0, z_vector_rot[2]], 'mo--', label='+Z axis (after Rx and Ry)')
        ax.legend()
        fig.tight_layout()

    return [rx,ry]



def plot_poses(df,extent = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    obj = ax.scatter(df['X'], df['Y'], df['Z'], c=df['color'], s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     kw = dict(prop="colors", num=df['color'].nunique(),
#           func=lambda s: df.index[s])
#     legend = ax.legend(*obj.legend_elements(**kw))
#     ax.add_artist(legend)
    ax.quiver(df['X'],df['Y'],df['Z'],df['uvec_X'],df['uvec_Y'],df['uvec_Z'],color='b',
              linewidth=1,arrow_length_ratio=0)
    ax.scatter(df['X']+df['uvec_X'], df['Y']+df['uvec_Y'], df['Z']+df['uvec_Z'], c=df['color'], s=50,marker='x')
    if extent is None:
        xsize = 700
        ysize = 700
        zsize = 700
        extent=[[df.loc['sMask','X']-xsize,df.loc['sMask','X']+xsize],
                      [df.loc['sMask','Y']-ysize,df.loc['sMask','Y']+ysize],
                      [df.loc['sMask','Z']-zsize,df.loc['sMask','Z']+zsize]]
    ax.axis('tight')
    ax.set_xlim3d(extent[0][0], extent[0][1])
    ax.set_ylim3d(extent[1][0], extent[1][1])
    ax.set_zlim3d(extent[2][0], extent[2][1])
    ax.view_init(30, 0, 90)
    fig.tight_layout()
    return ax

def plot_poses_no_extent(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    obj = ax.scatter(df['X'], df['Y'], df['Z'], c=df['color'], s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     kw = dict(prop="colors", num=df['color'].nunique(),
#           func=lambda s: df.index[s])
#     legend = ax.legend(*obj.legend_elements(**kw))
#     ax.add_artist(legend)
    ax.quiver(df['X'],df['Y'],df['Z'],df['uvec_X'],df['uvec_Y'],df['uvec_Z'],color='b',
              linewidth=1,arrow_length_ratio=0)
    ax.scatter(df['X']+df['uvec_X'], df['Y']+df['uvec_Y'], df['Z']+df['uvec_Z'], c=df['color'], s=50,marker='x')
    ax.axis('tight')
    # ax.set_xlim3d(extent[0][0], extent[0][1])
    # ax.set_ylim3d(extent[1][0], extent[1][1])
    # ax.set_zlim3d(extent[2][0], extent[2][1])
    ax.view_init(30, 0, 90)
    fig.tight_layout()
    return ax

def plot_whiskers(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # obj = ax.scatter(df['X'], df['Y'], df['Z'], c=df['color'], s=50)
    colorlist = plt.cm.Reds(np.linspace(0.1, 1, len(df)))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     kw = dict(prop="colors", num=df['color'].nunique(),
#           func=lambda s: df.index[s])
#     legend = ax.legend(*obj.legend_elements(**kw))
#     ax.add_artist(legend)
    ax.quiver(df['X']+df['uvec_X'], df['Y']+df['uvec_Y'], df['Z']+df['uvec_Z'],
            df['uvec_dx']-(df['X']+df['uvec_X']),df['uvec_dy']-(df['Y']+df['uvec_Y']),df['uvec_dz']-(df['Z']+df['uvec_Z']),color=colorlist,
            linewidth=1,arrow_length_ratio=0)
    ax.scatter(df['X']+df['uvec_X'], df['Y']+df['uvec_Y'], df['Z']+df['uvec_Z'], c=colorlist, s=50)
    ax.scatter(df['uvec_dx'],df['uvec_dy'],df['uvec_dz'], c=colorlist, s=50, marker='x')
    # ax.scatter(df['uvec_dx']-(df['X']+df['uvec_X']),df['uvec_dy']-(df['Y']+df['uvec_Y']),df['uvec_dz']-(df['Z']+df['uvec_Z']), c=df['color'], s=50,marker='x')
    ax.axis('tight')
    # ax.set_xlim3d(extent[0][0], extent[0][1])
    # ax.set_ylim3d(extent[1][0], extent[1][1])
    # ax.set_zlim3d(extent[2][0], extent[2][1])
    ax.view_init(30, 0, 90)
    fig.tight_layout()
    return ax


def plot_sMPA(df, ax):
    # X = df.loc['sMPA'],'X'] + np.arange(-200.0, 200.0, 5)
    # Y = df.loc['sMPA'],'Y'] + np.arange(-150.0, 80.0, 5)
    # X = np.arange(-180.0, 180.0, 5)
    # Y = np.arange(-120.0, 100.0, 5)
    # Z = 0
    # X,Y,Z = np.meshgrid(X, Y, Z)
    # xshape = X.shape
    # rotmat = R.from_euler('XYZ',sMPA_angle_to_WCS_deg, degrees=True)
    # val = rotmat.apply(np.array([X.ravel(),Y.ravel(),Z.ravel()]).T)
    # X = val[:,0].reshape(xshape)+ df.loc[['sMPA'],'X'].values.astype(float)
    # Y = val[:,1].reshape(xshape)+ df.loc[['sMPA'],'Y'].values.astype(float)
    # Z = val[:,2].reshape(xshape)+ df.loc[['sMPA'],'Z'].values.astype(float)
    # ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0], rstride=1, cstride=1, alpha=0.2)


    normal = df.loc['sMPA',['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
    center = df.loc['sMPA',['X','Y','Z']].values.astype(float)
    d = -center.dot(normal)
    X = np.arange(-180.0, 180.0, 5)+center[0]
    Y = np.arange(-120.0, 100.0, 5)+center[1]
    X,Y = np.meshgrid(X, Y)
    Z = (-normal[0] * X - normal[1] * Y - d) * 1. /normal[2]
    # X = val[:,0].reshape(xshape)+ df.loc[['sMPA'],'X'].values.astype(float)
    # Y = val[:,1].reshape(xshape)+ df.loc[['sMPA'],'Y'].values.astype(float)
    # Z = val[:,2].reshape(xshape)+ df.loc[['sMPA'],'Z'].values.astype(float)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

def plot_local_PR(df, ax, camera):
    # X = df.loc['sMPA'],'X'] + np.arange(-200.0, 200.0, 5)
    # Y = df.loc['sMPA'],'Y'] + np.arange(-150.0, 80.0, 5)
    # X = np.arange(-180.0, 180.0, 5)
    # Y = np.arange(-120.0, 100.0, 5)
    # Z = 0
    # X,Y,Z = np.meshgrid(X, Y, Z)
    # xshape = X.shape
    # rotmat = R.from_euler('XYZ',sMPA_angle_to_WCS_deg, degrees=True)
    # val = rotmat.apply(np.array([X.ravel(),Y.ravel(),Z.ravel()]).T)
    # X = val[:,0].reshape(xshape)+ df.loc[['sMPA'],'X'].values.astype(float)
    # Y = val[:,1].reshape(xshape)+ df.loc[['sMPA'],'Y'].values.astype(float)
    # Z = val[:,2].reshape(xshape)+ df.loc[['sMPA'],'Z'].values.astype(float)
    # ax.plot_surface(X[:,:,0], Y[:,:,0], Z[:,:,0], rstride=1, cstride=1, alpha=0.2)

    normal = df.loc['sMPA',['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
    rotmat = R.from_euler('XYZ',df.loc['sMPA',['Rx','Ry','Rz']].values.astype(float), degrees=True)
    center = df.loc['sMPA',['X','Y','Z']].values.astype(float)+rotmat.apply(PRcenters[camera])
    d = -center.dot(normal)
    X = np.arange(-0.1, 0.1, 0.01) + center[0]
    Y = np.arange(-0.1, 0.1, 0.01) + center[1]
    X,Y = np.meshgrid(X, Y)
    Z = (-normal[0] * X - normal[1] * Y - d) * 1. /normal[2]
    # X = val[:,0].reshape(xshape)+ df.loc[['sMPA'],'X'].values.astype(float)
    # Y = val[:,1].reshape(xshape)+ df.loc[['sMPA'],'Y'].values.astype(float)
    # Z = val[:,2].reshape(xshape)+ df.loc[['sMPA'],'Z'].values.astype(float)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)


def compute_endpoints(df,ignored_poses=[None],pose_select='PR'):
    newdf = pd.DataFrame()
    for pose in [val for val in df.index if pose_select in val and val not in ignored_poses]:
        newdf.loc[pose,'endpt_X'] = df.loc[pose,'X']+df.loc[pose,'uvec_X']
        newdf.loc[pose,'endpt_Y'] = df.loc[pose,'Y']+df.loc[pose,'uvec_Y']
        newdf.loc[pose,'endpt_Z'] = df.loc[pose,'Z']+df.loc[pose,'uvec_Z']
    return newdf

def compute_endpoints_with_FDPR_shifts(df,shifts_from_FDPR,pose_select='PR'):
    sMPA_rotmat = R.from_euler('XYZ',df.loc['sMPA',['Rx','Ry','Rz']].values.astype(float), degrees=True)
    endpoints = compute_endpoints(df)
    endpoints_with_shift = pd.DataFrame(columns=['endpt_X','endpt_Y','endpt_Z'])
    for pose in [val for val in df.index if pose_select in val]:
        dz = shifts_from_FDPR.loc[pose,['chief ray dz (um)']].values.astype(float)/1000.
        vec = df.loc[pose,['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
        normvec = vec/np.linalg.norm(vec)
        newvec1 = endpoints.loc[pose] - normvec*dz
        dxdy = np.array([shifts_from_FDPR.loc[pose,['dx (um)']][0],shifts_from_FDPR.loc[pose,['dy (um)']][0],0.])/1000.
        newvec2 = newvec1 - sMPA_rotmat.apply(dxdy)
        endpoints_with_shift.loc[pose] = newvec2
    return endpoints_with_shift

def compute_endpoint_errors(df,df1,pose_select='PR',ignored_poses=[None]):
    newdf = pd.DataFrame()
    newdf1 = pd.DataFrame()
    for pose in [val for val in df.index if pose_select in val and val not in ignored_poses]:
        newdf.loc[pose,'endpt_X'] = df.loc[pose,'X']+df.loc[pose,'uvec_X']
        newdf.loc[pose,'endpt_Y'] = df.loc[pose,'Y']+df.loc[pose,'uvec_Y']
        newdf.loc[pose,'endpt_Z'] = df.loc[pose,'Z']+df.loc[pose,'uvec_Z']
        newdf1.loc[pose,'endpt_X'] = df1.loc[pose,'X']+df1.loc[pose,'uvec_X']
        newdf1.loc[pose,'endpt_Y'] = df1.loc[pose,'Y']+df1.loc[pose,'uvec_Y']
        newdf1.loc[pose,'endpt_Z'] = df1.loc[pose,'Z']+df1.loc[pose,'uvec_Z']

    return newdf1-newdf

# turns out, not needed
def skew_matrix(v):
    return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

# turns out, not needed
def rotmat_from_2vec(a,b):
    anorm = np.linalg.norm(a)
    bnorm = np.linalg.norm(b)
    a_normalized = a/anorm
    b_normalized = b/bnorm
    v = np.cross(a_normalized,b_normalized)
    c = np.dot(a_normalized,b_normalized)
    vcross = skew_matrix(v)
    return np.eye(3)+vcross+np.dot(vcross,vcross)/(1.+c)

def compute_endpoint_errors_sMPA_frame(df, df1, pose_select='PR',ignored_poses=[None]):
    # express endpoint errors between 2 dataframes in the coordinate system of the sMPA in the second dataframe
    d = compute_endpoint_errors(df, df1,pose_select=pose_select,ignored_poses=ignored_poses)
    #d = compute_endpoints(df1)      #I don't THINK this will suffice as a drop-in solution
    newdf = pd.DataFrame(columns=d.columns)
    rotmat = R.from_matrix(rotmat_from_2vec(np.array([0,0,1]),df1.loc['sMPA',['uvec_X','uvec_Y','uvec_Z']].values.astype(float)))
    for pose in d.index:
        newdf.loc[pose] = rotmat.inv().apply(d.loc[pose].astype(float))
    return newdf

def compute_distance_to_sMPA(df,pose_select='PR'):
    normal = df.loc['sMPA',['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
    normal /= np.linalg.norm(normal)
    center = df.loc['sMPA',['X','Y','Z']].values.astype(float)
    ret = dict()
    for pose in [val for val in df.index if pose_select in val]:
        focus = df.loc[pose,['X','Y','Z']].values.astype(float)+df.loc[pose,['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
        dist = np.dot(focus-center,normal)
        ret[pose] = dist
    return ret

def compute_distance_to_PR_centers_Z(df,pose_select='PR'):
    normal = df.loc['sMPA',['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
    normal /= np.linalg.norm(normal)
    center = df.loc['sMPA',['X','Y','Z']].values.astype(float)
    ret = dict()
    for pose in [val for val in df.index if pose_select in val]:
        focus = df.loc[pose,['X','Y','Z']].values.astype(float)+df.loc[pose,['uvec_X','uvec_Y','uvec_Z']].values.astype(float)
        dist = np.dot(focus-center,normal)-PRcenters[pose][-1]
        ret[pose] = dist
    return ret


def check_pupil_crossing(dflist,ignored_poses=[None]):
    # print('Pose | Distance | ')
    for d in dflist:
        dfnew = pd.DataFrame(columns=['sMask to chief ray (mm)','sMask to origin (mm)'])
        for pose in [val for val in d.index if val not in ignored_poses]:
            vec = np.array(d.loc['sMask',['X','Y','Z']]-d.loc[pose,['X','Y','Z']]).astype(float)
            uvec = np.array(d.loc[pose,['uvec_X','uvec_Y','uvec_Z']]).astype(float)
            dist = np.linalg.norm(np.cross(vec,uvec))/np.linalg.norm(uvec)
            dfnew.loc[pose] = [dist,np.linalg.norm(d.loc['sMask',['X','Y','Z']]-d.loc[pose,['X','Y','Z']])]
        print (dfnew)

#Read in the relevant data from Manal's encoder transformation spreadsheet. You'll need to update the filepath below to your personal machine.
def read_encoder_coeffs_from_file():
    # encoder_converter_filepath = 'New Encoder Decoder 03042023.xlsx'
    encoder_converter_filepath = 'files/Encoder Mapping.xlsx'
    spreadsheet = pd.read_excel(encoder_converter_filepath,sheet_name='Encoder Mapping 5-18-2023',skiprows=3,usecols='H:M')

    coeffs_encoder_to_5DOF = spreadsheet.loc[0:20]
    coeffs_encoder_to_5DOF = coeffs_encoder_to_5DOF.set_index('Encoder Coefficient')
    coeffs_5DOF_to_encoder = spreadsheet.loc[27:47]
    coeffs_5DOF_to_encoder = coeffs_5DOF_to_encoder.set_index('Encoder Coefficient')
    return [coeffs_encoder_to_5DOF,coeffs_5DOF_to_encoder]

def calculate_5DOF_from_encoders(current_encoder_values):
    # print(current_encoder_values.loc['X'])
    terms = [current_encoder_values['X'],current_encoder_values['Y'],current_encoder_values['Z'],1,\
             current_encoder_values['X']**2,current_encoder_values['Y']**2,current_encoder_values['Z']**2,\
             current_encoder_values['X']**3,current_encoder_values['Y']**3,current_encoder_values['Z']**3,\
             current_encoder_values['X']*current_encoder_values['Y'],current_encoder_values['Z']*current_encoder_values['Y'],current_encoder_values['X']*current_encoder_values['Z'],\
             current_encoder_values['Rx'],current_encoder_values['Ry'],1,\
             current_encoder_values['Rx']**2,current_encoder_values['Ry']**2,\
             current_encoder_values['Rx']**3,current_encoder_values['Ry']**3,current_encoder_values['Rx']*current_encoder_values['Ry']]
    terms = [float(i) for i in terms]                               #Convert to a list of floats instead of a list of Series
    coeffs_encoder_to_5DOF = read_encoder_coeffs_from_file()[0]     #Read the transformation coefficient values from the excel file
    #Create an empty dataframe. Calculate the 5DOF position one component at a time by multiplying the "term" array with each column of the coefficient array.
    calculated_5DOF_values = pd.DataFrame([np.repeat(np.nan,5)],columns=coeffs_encoder_to_5DOF.columns,index=[current_encoder_values.index[0]])
    for column in coeffs_encoder_to_5DOF.columns:
        calculated_5DOF_values[column] = sum(coeffs_encoder_to_5DOF[column]*terms)

    return calculated_5DOF_values

def calculate_encoders_from_5DOF(current_5DOF_values):
    terms = [current_5DOF_values['X'],current_5DOF_values['Y'],current_5DOF_values['Z'],1,\
             current_5DOF_values['X']**2,current_5DOF_values['Y']**2,current_5DOF_values['Z']**2,\
             current_5DOF_values['X']**3,current_5DOF_values['Y']**3,current_5DOF_values['Z']**3,\
             current_5DOF_values['X']*current_5DOF_values['Y'],current_5DOF_values['Z']*current_5DOF_values['Y'],current_5DOF_values['X']*current_5DOF_values['Z'],\
             current_5DOF_values['Rx'],current_5DOF_values['Ry'],1,\
             current_5DOF_values['Rx']**2,current_5DOF_values['Ry']**2,\
             current_5DOF_values['Rx']**3,current_5DOF_values['Ry']**3,current_5DOF_values['Rx']*current_5DOF_values['Ry']]
    terms = [float(i) for i in terms]                               #Convert to a list of floats instead of a list of Series
    coeffs_encoder_to_5DOF = read_encoder_coeffs_from_file()[1]     #Read the transformation coefficient values from the excel file

    #Create an empty dataframe. Calculate the 5DOF position one component at a time by multiplying the "term" array with each column of the coefficient array.
    calculated_encoder_values = pd.DataFrame([np.repeat(np.nan,5)],columns=coeffs_encoder_to_5DOF.columns,index=[current_5DOF_values.index[0]])
    for column in coeffs_encoder_to_5DOF.columns:
        calculated_encoder_values[column] = sum(coeffs_encoder_to_5DOF[column]*terms)

    return calculated_encoder_values


# For convenience use this function
# https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(Rmat) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        Rmat = Vt.T @ U.T

    t = -Rmat @ centroid_A + centroid_B

    return Rmat, t

def translate_rotate_poses(df,rot,t):
    dfnew = pd.DataFrame(columns=df.columns)
    rotmat = R.from_matrix(rot)
    rotmat_euler = rotmat.as_euler('XYZ',degrees=True)
    print(f'Euler XYZ (deg); Z will be ignored: {rotmat_euler}')
    for pose in df.index:
        dfnew.loc[pose,['X','Y','Z']] = rotmat.apply(df.loc[pose,['X','Y','Z']].astype(float))+t.flatten()
        dfnew.loc[pose,['uvec_X','uvec_Y','uvec_Z']] = rotmat.apply(df.loc[pose,['uvec_X','uvec_Y','uvec_Z']].astype(float)) #np.dot(update_rot.as_matrix(),df_update.loc[pose,['uvec_X','uvec_Y','uvec_Z']])
        dfnew.loc[pose,['Rx','Ry']] = df.loc[pose,['Rx','Ry']].astype(float) + np.array(rotmat_euler[:-1])
    dfnew['color'] = df['color']
    return dfnew


def modify_poses_5DOF(dp, df,focal_length,GSA_angle_WCS_deg,pose_select=None):
    newdf = pd.DataFrame(columns=df.columns)
    if pose_select is not None:
        poses_to_modify = [val for val in df.index if pose_select in val]
    else:
        poses_to_modify = [val for val in df.index if ('PR' in val) or ('PD' in val)]
    for pose in poses_to_modify:
        newdf.loc[pose,['X','Y','Z','Rx','Ry']] = df.loc[pose,['X','Y','Z','Rx','Ry']].astype(float) + np.array(dp)
        update_uvec(newdf,pose,focal_length,GSA_angle_WCS_deg)
    return newdf

def modify_poses_6DOF(delta_poses, df,focal_length,GSA_angle_WCS_deg,pose_select=None):
    newdf = pd.DataFrame(columns=df.columns)
    if pose_select is not None:
        poses_to_modify = [val for val in df.index if pose_select in val]
    else:
        poses_to_modify = [val for val in df.index if ('PR' in val) or ('PD' in val)]
    for pose in poses_to_modify:
        #Apply the 5DOF tweak and recalculate uvecs
        # print(df.loc[pose,['uvec_X','uvec_Y','uvec_Z']])
        newdf.loc[pose,['X','Y','Z','Rx','Ry']] = df.loc[pose,['X','Y','Z','Rx','Ry']].astype(float) + np.array(delta_poses[0:-1])
        update_uvec(newdf,pose,focal_length,GSA_angle_WCS_deg)
        # print(newdf.loc[pose,['uvec_X','uvec_Y','uvec_Z']])

        #Rotate the uvecs about Z
        newdf.loc[pose,['uvec_X','uvec_Y','uvec_Z']] = rotate_pose_about_Rz(newdf.loc[pose,['uvec_X','uvec_Y','uvec_Z']],delta_poses[-1])
        # print(newdf.loc[pose,['uvec_X','uvec_Y','uvec_Z']])

        #Recalculate Rx/Ry based on updated uvecs
        update_RxRy(newdf,pose,GSA_angle_WCS_deg)

    return newdf

def rotate_pose_about_Rz(pose_vector,Rz):
    rot_Rz = R.from_euler('Z',Rz,degrees=True)
    new_vector = rot_Rz.apply(np.array(pose_vector).astype(float))
    return new_vector

#This function is to be used exclusively with the curve_fit.
#There are better ways to calculate endpoints if you're doing it "by hand".
#Recall that all curve_fit inputs/outputs must be 1D arrays, NOT DataFrames.
def generate_endpoints_for_fitting(df,dx,dy,dz,drx,dry,focal_length,GSA_angle_WCS_deg,translation_to_sMPA,
                                   rotation_from_sMPA_to_5DOF,pose_select='PR'):
    df_temp = modify_poses_5DOF([dx,dy,dz,drx,dry],df,focal_length,GSA_angle_WCS_deg)
    endpoints_temp_5DOF = compute_endpoints(df_temp)    #With the current shifts, calculate the endpoints in the 5DOF frame

    return endpoints_temp_5DOF.values.ravel() #We have to flatten the array so that curve_fit doesn't throw a fit. We'll reshape it later.

def generate_endpoints_for_fitting_6DOF(df,dx,dy,dz,drx,dry,drz,focal_length,GSA_angle_WCS_deg,translation_to_sMPA,
                                   rotation_from_sMPA_to_5DOF,pose_select='PR'):
    df_temp = modify_poses_6DOF([dx,dy,dz,drx,dry,drz],df,focal_length,GSA_angle_WCS_deg)
    endpoints_temp_6DOF = compute_endpoints(df_temp)    #With the current shifts, calculate the endpoints in the 5DOF frame

    return endpoints_temp_6DOF.values.ravel() #We have to flatten the array so that curve_fit doesn't throw a fit. We'll reshape it later.

def convert_endpoints_to_sMPA_frame(df_endpoints_5DOF,translation_to_sMPA,rotation_from_sMPA_to_5DOF,pose_select='PR'):
    df_endpoints_sMPA = df_endpoints_5DOF.copy()
    #print('Endpoints in 5DOF CS: \n',df_endpoints_5DOF,'\n')

    #Start by translating the endpoints to the sMPA CS
    for pose in [val for val in df_endpoints_5DOF.index if pose_select in val]:
        df_endpoints_sMPA.loc[pose] -= translation_to_sMPA.values.astype(float)
    #print('Endpoints in sMPA CS (pre-rotation): \n',df_endpoints_sMPA,'\n')

    #Then rotate the endpoints into the sMPA CS
    for pose in [val for val in df_endpoints_5DOF.index if pose_select in val]:
        df_endpoints_sMPA.loc[pose] = rotation_from_sMPA_to_5DOF.inv().apply(df_endpoints_sMPA.loc[pose].values.astype(float))
    #print('Endpoints in sMPA CS (post-rotation): \n',df_endpoints_sMPA)

    return df_endpoints_sMPA

def pose_update_with_FDPR_results(df,shifts_from_FDPR,focal_length,GSA_angle_WCS_deg,translation_to_sMPA,rotation_from_sMPA_to_5DOF):
    endpoints_5DOF = compute_endpoints(df)
    #endpoints_sMPA = convert_endpoints_to_sMPA_frame(endpoints_5DOF,translation_to_sMPA,rotation_from_sMPA_to_5DOF)

    #Update the endpoints based on information from the FDPR team
    endpoints_nominal = compute_endpoints_with_FDPR_shifts(df, shifts_from_FDPR)
    #endpoints_nominal = endpoints_5DOF.copy()
    #for pose in [pose for pose in endpoints_nominal.index if 'PR' in pose]:
    #    endpoints_nominal.loc[pose] = create_error_vector_v2(endpoints_nominal.loc[pose],shifts_from_FDPR)
        
        #endpoints_sMPA_nominal.loc[pose] -= shifts_from_FDPR.loc[pose,['dx (um)','dy (um)','chief ray dz (um)']].values/1000
    #print('FDPR-adjusted nominal endpoint locations (sMPA frame): \n',endpoints_sMPA_nominal,'\n')

    #Initial guesses/bounds for translations and rotations in millimeters and degrees
    initial_guesses = [0.,0.,0.,0.,0.]
    bounds = ((-10,-10,-10,-1,-1),
              (10,10,10,1,1))

    endpoints_residuals_baseline = endpoints_nominal - endpoints_5DOF
    
    print('Residuals (in sMPA frame) baseline: \n',endpoints_residuals_baseline,'\n')

    #Important caveat here, curve_fit only handles 1D arrays.
    #So we have to strip our dataframe down to a 2D array and then flatten it, and reshape it later.
    best_fit_deltas, best_fit_errors = curve_fit(partial(generate_endpoints_for_fitting,focal_length=focal_length,
                                                GSA_angle_WCS_deg=GSA_angle_WCS_deg,
                                                translation_to_sMPA=translation_to_sMPA,
                                                rotation_from_sMPA_to_5DOF=rotation_from_sMPA_to_5DOF),
                                                df,endpoints_nominal.values.ravel(),p0=initial_guesses,bounds=bounds)

    print('Best-fit (X,Y,Z,Rx,Ry) deltas to apply to real-space poses: \n',np.round(best_fit_deltas,4),'\n')

    #Update the poses in the 5DOF frame
    df_after_fitting_FDPR_shifts = modify_poses_5DOF(best_fit_deltas,df,focal_length,GSA_angle_WCS_deg)
    #poses_to_display = [val for val in df.index if ('PR' in val) or ('PD' in val)]
    #print('Old poses (in real space): \n',df.loc[poses_to_display,['X','Y','Z','Rx','Ry']],'\n')
    #print('New poses (in real space): \n',df_after_fitting_FDPR_shifts.loc[poses_to_display,['X','Y','Z','Rx','Ry']],'\n')

    #Calculate the endpoints of this new best-fit set of poses within the sMPA frame
    endpoints_best_fit = compute_endpoints(df_after_fitting_FDPR_shifts)
    #print('Best-fit endpoints (sMPA frame): \n',endpoints_sMPA_best_fit,'\n')

    #Calculate residuals between nominal and best-fit
    endpoints_residuals = endpoints_nominal - endpoints_best_fit
    print('Residuals (in sMPA frame) after incorporating the best-fit (x,y,z,Rx,Ry) shift to all poses: \n',np.round(endpoints_residuals,3),'\n')

    return df_after_fitting_FDPR_shifts,endpoints_residuals,best_fit_deltas,best_fit_errors

def pose_update_with_FDPR_results_v2(df,shifts_from_FDPR,focal_length,GSA_angle_WCS_deg,translation_to_sMPA,rotation_from_sMPA_to_5DOF):
    endpoints_5DOF = compute_endpoints(df)
    #endpoints_sMPA = convert_endpoints_to_sMPA_frame(endpoints_5DOF,translation_to_sMPA,rotation_from_sMPA_to_5DOF)

    #Update the endpoints based on information from the FDPR team
    endpoints_nominal = compute_endpoints_with_FDPR_shifts(df, shifts_from_FDPR)
    #endpoints_nominal = endpoints_5DOF.copy()
    #for pose in [pose for pose in endpoints_nominal.index if 'PR' in pose]:
    #    endpoints_nominal.loc[pose] = create_error_vector_v2(endpoints_nominal.loc[pose],shifts_from_FDPR)
        
        #endpoints_sMPA_nominal.loc[pose] -= shifts_from_FDPR.loc[pose,['dx (um)','dy (um)','chief ray dz (um)']].values/1000
    #print('FDPR-adjusted nominal endpoint locations (sMPA frame): \n',endpoints_sMPA_nominal,'\n')

    #Initial guesses/bounds for translations and rotations in millimeters and degrees
    initial_guesses = [0.,0.,0.,0.,0.,0.]
    bounds = ((-10,-10,-10,-1,-1,-5),
              (10,10,10,1,1,5))

    endpoints_residuals_baseline = endpoints_nominal - endpoints_5DOF
    print('Residuals baseline: \n',endpoints_residuals_baseline,'\n')

    #Important caveat here, curve_fit only handles 1D arrays.
    #So we have to strip our dataframe down to a 2D array and then flatten it, and reshape it later.
    best_fit_deltas, best_fit_errors = curve_fit(partial(generate_endpoints_for_fitting_6DOF,focal_length=focal_length,
                                                GSA_angle_WCS_deg=GSA_angle_WCS_deg,
                                                translation_to_sMPA=translation_to_sMPA,
                                                rotation_from_sMPA_to_5DOF=rotation_from_sMPA_to_5DOF),
                                                df,endpoints_nominal.values.ravel(),p0=initial_guesses,bounds=bounds)

    print('Best-fit 6DoF deltas to apply to real-space poses: \n',np.round(best_fit_deltas,4),'\n')

    #Update the poses in the 5DOF frame
    df_after_fitting_FDPR_shifts = modify_poses_6DOF(best_fit_deltas,df,focal_length,GSA_angle_WCS_deg)
    #poses_to_display = [val for val in df.index if ('PR' in val) or ('PD' in val)]
    #print('Old poses (in real space): \n',df.loc[poses_to_display,['X','Y','Z','Rx','Ry']],'\n')
    #print('New poses (in real space): \n',df_after_fitting_FDPR_shifts.loc[poses_to_display,['X','Y','Z','Rx','Ry']],'\n')

    #Calculate the endpoints of this new best-fit set of poses within the sMPA frame
    endpoints_best_fit = compute_endpoints(df_after_fitting_FDPR_shifts)
    #print('Best-fit endpoints (sMPA frame): \n',endpoints_sMPA_best_fit,'\n')

    #Calculate residuals between nominal and best-fit
    endpoints_residuals = endpoints_nominal - endpoints_best_fit
    print('Residuals after incorporating the best-fit 6DoF shift to all poses: \n',np.round(endpoints_residuals,3),'\n')

    return df_after_fitting_FDPR_shifts,endpoints_residuals,best_fit_deltas,best_fit_errors

def optimize_5DOF_rotation_for_PAT(current_AC_AZ, current_AC_EL, desired_AC_AZ, desired_AC_EL, current_GSARX, current_GSARY,print_details=False):
    transformation_matrix_az_el = np.array([[0.824126, -0.56641],
                                            [0.566407, 0.824126]])
    optimal_GSARX,optimal_GSARY = np.array([current_GSARX,current_GSARY]) +\
        np.matmul(np.array([desired_AC_AZ,desired_AC_EL]) - np.array([current_AC_AZ,current_AC_EL]),transformation_matrix_az_el)
    if print_details == True:
        print('Starting GSARX/RY: ',current_GSARX, current_GSARY)
        print('Optimal GSARX/RY: ',optimal_GSARX,optimal_GSARY)
    return optimal_GSARX,optimal_GSARY

def optimize_5DOF_translation_for_PAT(current_X_CENTR, current_Y_CENTR, desired_X_CENTR, desired_Y_CENTR, current_HTSAX, current_VTSA,print_details=False):
    transformation_matrix_x_y = np.array([[3.853e-2,-2.897e-2],
                                          [-2.953e-2,-3.862e-2]])
    optimal_HTSAX,optimal_VTSA = np.array([current_HTSAX,current_VTSA]) +\
        np.matmul(np.array([desired_X_CENTR,desired_Y_CENTR]) - np.array([current_X_CENTR, current_Y_CENTR]),transformation_matrix_x_y)
    if print_details == True:
        print('Starting HTSA/VTSA: ',current_HTSAX, current_VTSA)
        print('Optimal HTSA/VTSA: ',optimal_HTSAX,optimal_VTSA)
    return optimal_HTSAX,optimal_VTSA

def convert_pose_encoders_to_pose_actual(pose_encoders):
    pose_actual = pd.DataFrame(columns=pose_encoders.columns)
    for pose in [val for val in pose_encoders.index if ('PR' in val) or ('PD' in val)]:
        pose_actual = pd.concat([pose_actual,calculate_5DOF_from_encoders(pd.DataFrame(pose_encoders.loc[pose]).T)])
    # print('NEWLY calculated 5DOF position in real space based on recorded 5DOF encoder values: \n',pose_actual)
    return pose_actual

def convert_df_to_encoder_space(df,pose_select=None,ignored_poses=[None]):
    df_encoder_space = pd.DataFrame(columns=['X', 'Y', 'Z', 'Rx', 'Ry'])
    if pose_select == None:
        poses_to_convert = [val for val in df.index if 'PDI' in val or 'PR' in val and val not in ignored_poses]
    else:
        poses_to_convert = [val for val in df.index if pose_select in val and val not in ignored_poses]
    for pose in poses_to_convert:
        df_encoder_space = pd.concat([df_encoder_space,calculate_encoders_from_5DOF(pd.DataFrame(df.loc[pose]).T)])
    return df_encoder_space

def optimize_p_null_PAT_encoders(df_PAT_encoders,p_null_offset):
    optimal_gsarx_PAT, optimal_gsary_PAT = optimize_5DOF_rotation_for_PAT(df_PAT_encoders['PAT AC AZ'].values[0], df_PAT_encoders['PAT AC EL'].values[0], 
                                         df_PAT_encoders['s/MATF AC AZ'].values[0],df_PAT_encoders['s/MATF AC EL'].values[0], 
                                         df_PAT_encoders['Rx'].values[0], df_PAT_encoders['Ry'].values[0],print_details=True)

    optimal_htsax_PAT, optimal_vtsa_PAT = optimize_5DOF_translation_for_PAT(df_PAT_encoders['PAT LED X'].values[0], df_PAT_encoders['PAT LED Y'].values[0], 
                                             df_PAT_encoders['s/MATF LED X'].values[0],df_PAT_encoders['s/MATF LED Y'].values[0], 
                                             df_PAT_encoders['X'].values[0], df_PAT_encoders['Y'].values[0],print_details=True)
    
    additional_PAT_z_shift = -1*p_null_offset[1]*np.sin(np.deg2rad(optimal_gsarx_PAT - df_PAT_encoders['Rx'].values[0])) +\
                                p_null_offset[0]*np.sin(np.deg2rad(optimal_gsary_PAT - df_PAT_encoders['Ry'].values[0]))
    #print('Effective change in Z-position of PAT after applying optimal rotations: ',round(additional_PAT_z_shift,4))
    
    #Update the dataframe with the new, optimal values.
    df_PAT_encoders['z_PAT'] += additional_PAT_z_shift
    df_PAT_encoders[['X','Y','Rx','Ry']] = [optimal_htsax_PAT,optimal_vtsa_PAT,optimal_gsarx_PAT,optimal_gsary_PAT]
    df_PAT_encoders[['PAT AC AZ','PAT AC EL','PAT LED X','PAT LED Y']] = df_PAT_encoders[['s/MATF AC AZ','s/MATF AC EL','s/MATF LED X','s/MATF LED Y']]
    df_PAT_encoders[['Optimized?']] = True
    
    return df_PAT_encoders

def get_data_from_ADM_log(plateau,z_type,index_name,pat_target,pat_led_type,matf_led_type,filepath = 'files/ADM Ops Log.xlsx',print_details=False):
    spreadsheet = pd.read_excel(filepath,sheet_name='Position Log - Ball',skiprows=0,usecols='A:Z',index_col=5)
    spreadsheet = spreadsheet[(spreadsheet['Plateau'] == plateau) & (spreadsheet['Final?'] == 'Y')]

    if 'sMATF mirror' in spreadsheet.index:
        matf_target = 'sMATF'
    elif 'MATF mirror' in spreadsheet.index:
        matf_target = 'MATF'    

    if z_type == 'retro':
        z_type_temp = 'retro pri'
    elif z_type == 'mirror':
        z_type_temp = 'mirror'
    df_parsed_from_ADMLog = pd.DataFrame(dict({'X':spreadsheet.loc['PAT'+pat_target+' LED '+pat_led_type]['5DOF X'], 'Y':spreadsheet.loc['PAT'+pat_target+' LED '+pat_led_type]['5DOF Y'],'Z':spreadsheet.loc['PAT'+pat_target+' LED pri']['5DOF Z'],
                                                  'Rx':spreadsheet.loc['PAT'+pat_target+' LED '+pat_led_type]['5DOF Rx'],'Ry':spreadsheet.loc['PAT'+pat_target+' LED '+pat_led_type]['5DOF Ry'],
                                                  'z_s/MATF':spreadsheet.loc[matf_target+' '+z_type]['Range (m)']*1000.,
                                                  'z_PAT':spreadsheet.loc['PAT'+pat_target+' '+z_type_temp]['Range (m)']*1000.,
                                                  'z_type':z_type,
                                                  's/MATF AC AZ':spreadsheet.loc[matf_target+' mirror']['AC AZ (deg)'],
                                                  's/MATF AC EL':spreadsheet.loc[matf_target+' mirror']['AC EL (deg)'],
                                                  'PAT AC AZ':spreadsheet.loc['PAT'+pat_target+' mirror']['AC AZ (deg)'],
                                                  'PAT AC EL':spreadsheet.loc['PAT'+pat_target+' mirror']['AC EL (deg)'],
                                                  's/MATF LED X':spreadsheet.loc[matf_target+' LED '+matf_led_type]['X centr (pix)'],
                                                  's/MATF LED Y':spreadsheet.loc[matf_target+' LED '+matf_led_type]['Y centr (pix)'],
                                                  'PAT LED X':spreadsheet.loc['PAT'+pat_target+' LED '+pat_led_type]['X centr (pix)'],
                                                  'PAT LED Y':spreadsheet.loc['PAT'+pat_target+' LED '+pat_led_type]['Y centr (pix)'],
                                                  'date':spreadsheet.loc[matf_target+' mirror']['Date']
                                                 }),index=[index_name])
    if print_details == True:
        print(df_parsed_from_ADMLog.squeeze())
    return df_parsed_from_ADMLog

def write_new_poses_to_Excel(filename,new_pose_name,update_type,baseline_filepath,
                             columns,sMPA_angle_to_WCS_deg,GSA_angle_WCS_deg,df,df_encoders,
                             df_update,df_update_encoders,focal_length,num_poses,
                             baseline_ADM_plateau_name=None,update_ADM_plateau_name=None,
                             p_null_PAT_baseline_encoder_original=None,
                             p_null_PAT_update_encoder_original=None,
                             rigid_body_correction=None,best_fit_errors=None,ignored_poses=['None']):
    
    charstr='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chars=list(charstr)
    #nums=[str(i) for i in range(1,26)]

    startrow = 15
    startcol = 1
    index_names = [val for val in df.index if val not in ignored_poses]

    gsa_rot = R.from_euler('X',[GSA_angle_WCS_deg], degrees=True)

    filename = filename
    sheet1_name = 'baseline'
    sheet2_name = 'update'
    with pd.ExcelWriter(filename) as writer:  
        df.loc[index_names,['X','Y','Z','Rx','Ry']].to_excel(writer, sheet_name=sheet1_name,startrow=startrow,startcol=startcol)
        df_encoders.loc[index_names,['X','Y','Z','Rx','Ry']].to_excel(writer, sheet_name=sheet1_name,startrow=startrow,startcol=startcol+7)   
        sheet = writer.sheets[sheet1_name]
        sheet[f'{chars[startcol+7]}{startrow}']='Encoders'
        sheet[f'{chars[startcol+8]}{startrow}']='from '+baseline_filepath
        sheet[f'{chars[startcol]}{startrow}']='Position in 5DOF space'
        sheet['B1'] = new_pose_name
        # sheet[f'{chars[startcol]}{startrow+1}'] = 'Name'
        # sheet[f'{chars[startcol+12]}{startrow+1}'] = 'Name'
        sheet['A'+str(startrow+num_poses+3)] = 'Corresponding field points in GSA'
        sheet['A3'] = 'Track length'
        sheet['B3'] = focal_length+110
    #     sheet['A6'] = 'sMPA offset to WCS'
        newdf = df[df.index=='sMPA'][['X','Y','Z','uvec_X','uvec_Y','uvec_Z']]
        newdf.to_excel(writer,sheet_name=sheet1_name,startrow=4,startcol=0)
        # newdf = sMPA_angle_to_WCS_deg.to_frame().transpose()
        # newdf.columns = ['Rx','Ry','Rz']
        # newdf.to_excel(writer,sheet_name=sheet1_name,startrow=7,startcol=0)
        sheet['A10'] = 'GSA angle to WCS'
        sheet['B10'] = GSA_angle_WCS_deg
        if 'sMask' in df.index:
            newdf = df[df.index=='sMask'][['X','Y','Z']]
            newdf.loc['sMask',['X','Y','Z']] = gsa_rot.inv().apply(newdf.loc['sMask',['X','Y','Z']].values.astype(float))
            newdf.to_excel(writer,sheet_name=sheet1_name,startrow=11,startcol=1)
        endpoints = compute_endpoints(df.loc[index_names],pose_select='')
        gsa_coordinates = pd.DataFrame(gsa_rot.inv().apply(endpoints),columns=['X','Y','Z'])
        gsa_coordinates.index = index_names
        gsa_coordinates.to_excel(writer,sheet_name=sheet1_name,startrow=startrow+num_poses+3,startcol=1)

        if p_null_PAT_baseline_encoder_original is not None:
            sheet['B'+str(startrow+num_poses*2+6)] = 'ADM measurements'
            sheet['D'+str(startrow+num_poses*2+6)] = 'from "'+baseline_ADM_plateau_name+'"'
            p_null_PAT_baseline_encoder_original.to_excel(writer, sheet_name=sheet1_name,startrow=startrow+num_poses*2+6,startcol=1)

        #Writing updated pose info to the 2nd tab      
        df_update.loc[index_names,['X','Y','Z','Rx','Ry']].to_excel(writer, sheet_name=sheet2_name,startrow=startrow,startcol=startcol)
        df_update_encoders.loc[index_names,['X','Y','Z','Rx','Ry']].to_excel(writer, sheet_name=sheet2_name,startrow=startrow,startcol=startcol+7)   
        sheet = writer.sheets[sheet2_name]
        sheet[f'{chars[startcol+7]}{startrow}']='Encoders'
        sheet[f'{chars[startcol]}{startrow}']='Position in 5DOF space'
        sheet['B1'] = new_pose_name
        sheet[f'{chars[startcol]}{startrow+1}'] = 'Name'
        sheet[f'{chars[startcol+7]}{startrow+1}'] = 'Name'
        sheet['A'+str(startrow+num_poses+3)] = 'Corresponding field points in GSA'
        sheet['A3'] = 'Track length'
        sheet['B3'] = focal_length+110
    #     sheet['A6'] = 'sMPA offset to WCS'
        # newdf = df_update[df_update.index=='sMPA'][['X','Y','Z','uvec_X','uvec_Y','uvec_Z']]
        # newdf.to_excel(writer,sheet_name=sheet2_name,startrow=4,startcol=0)
        # newdf = sMPA_angle_to_WCS_deg.to_frame().transpose()
        # newdf.columns = ['Rx','Ry','Rz']
        # newdf.to_excel(writer,sheet_name=sheet2_name,startrow=7,startcol=0)
        sheet['A10'] = 'GSA angle to WCS'
        sheet['B10'] = GSA_angle_WCS_deg
        if 'sMask' in df_update.index:
            newdf = df_update[df_update.index=='sMask'][['X','Y','Z']]
            newdf.loc['sMask',['X','Y','Z']] = gsa_rot.inv().apply(newdf.loc['sMask',['X','Y','Z']].values.astype(float))
            newdf.to_excel(writer,sheet_name=sheet2_name,startrow=11,startcol=1)
        endpoints = compute_endpoints(df_update.loc[index_names],pose_select='')
        gsa_coordinates = pd.DataFrame(gsa_rot.inv().apply(endpoints),columns=['X','Y','Z'])
        gsa_coordinates.index = index_names
        gsa_coordinates.to_excel(writer,sheet_name=sheet2_name,startrow=startrow+num_poses+3,startcol=1)

        if p_null_PAT_update_encoder_original is not None:
            sheet['B'+str(startrow+num_poses*2+6)] = 'ADM measurements'
            sheet['D'+str(startrow+num_poses*2+6)] = 'from "'+update_ADM_plateau_name+'"'
            p_null_PAT_update_encoder_original.to_excel(writer, sheet_name=sheet2_name,startrow=startrow+num_poses*2+6,startcol=1)
            
        if update_type == 'FDPR' and rigid_body_correction is not None:
            newdf = pd.DataFrame(columns=['X','Y','Z','Rx','Ry','Rz'],index=['rigid body transform'])
            newdf.loc['rigid body transform'] = rigid_body_correction
            newdf.to_excel(writer,sheet_name=sheet2_name,startrow=startrow,startcol=13)
            sheet['I'+str(startrow+num_poses+3)] = 'Best-fit rigid body transform:'
            
            newdf = pd.DataFrame(best_fit_errors,columns=['X','Y','Z','Rx','Ry','Rz'],index=['X','Y','Z','Rx','Ry','Rz'])
            newdf.to_excel(writer,sheet_name=sheet2_name,startrow=startrow+num_poses+3,startcol=13)
            sheet['I'+str(startrow+num_poses+8)] = 'Covariance matrix from fit:'

        
        print('**Writing to Excel complete.**')
        print('**Filename: ',filename)

