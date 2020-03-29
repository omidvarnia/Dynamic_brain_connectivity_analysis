function varargout = subject_level_MRI_stats(varargin)
% SUBJECT_LEVEL_MRI_STATS MATLAB code for subject_level_MRI_stats.fig
%      SUBJECT_LEVEL_MRI_STATS, by itself, creates a new SUBJECT_LEVEL_MRI_STATS or raises the existing
%      singleton*.
%
%      H = SUBJECT_LEVEL_MRI_STATS returns the handle to a new SUBJECT_LEVEL_MRI_STATS or the handle to
%      the existing singleton*.
%
%      SUBJECT_LEVEL_MRI_STATS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SUBJECT_LEVEL_MRI_STATS.M with the given input arguments.
%
%      SUBJECT_LEVEL_MRI_STATS('Property','Value',...) creates a new SUBJECT_LEVEL_MRI_STATS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before subject_level_MRI_stats_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to subject_level_MRI_stats_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help subject_level_MRI_stats

% Last Modified by GUIDE v2.5 31-Oct-2019 13:24:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @subject_level_MRI_stats_OpeningFcn, ...
    'gui_OutputFcn',  @subject_level_MRI_stats_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before subject_level_MRI_stats is made visible.
function subject_level_MRI_stats_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to subject_level_MRI_stats (see VARARGIN)

% Choose default command line output for subject_level_MRI_stats
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes subject_level_MRI_stats wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = subject_level_MRI_stats_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton1,'FontName','default')
ni = questdlg('Do you want to calculate local connectivity (ReHo)?', ...
    'Select option', ...
    'Yes','No',[1 50]);

switch ni
    case 'Yes'
        [clinical_file, clinical_path] = uigetfile('*.nii','Select pre_processed_rsfMRI.nii 4D file - clinical subject');
        handles.clinical = [clinical_path clinical_file];
        guidata(hObject,handles);
        handles.resp = ni;
        guidata(hObject,handles);
        
    case 'No'
        [clinical_file, clinical_path] = uigetfile('*.nii','Select your_own_MRI_metric.nii 3D file - clinical subject');
        handles.clinical = [clinical_path clinical_file];
        guidata(hObject,handles);
        handles.resp = ni;
        guidata(hObject,handles);
        
end

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton2,'FontName','default')
ni = questdlg('Do you want to calculate local connectivity (ReHo)?', ...
    'Select option', ...
    'Yes','No',[1 50]);

switch ni
    case 'Yes'
        handles.controls_path = uigetdir('Select several pre_processed_rsfMRI.nii 4D files - control subjects');
        guidata(hObject,handles);
        handles.resp = ni;
        guidata(hObject,handles);
        
    case 'No'
        handles.controls_path = uigetdir('Select several pyour_own_MRI_metric.nii 3D files - control subjects');
        guidata(hObject,handles);
        handles.resp = ni;
        guidata(hObject,handles);
        
end

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton3,'FontName','default')
[mask_file,mask_path] = uigetfile('*.nii','Select single *.nii file - e..g, a gray matter mask');
handles.mask = [mask_path mask_file];
guidata(hObject,handles);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton4,'FontName','default')
x = inputdlg('Smoothing (~6-8 mm is often a good estimate ...):',...
    'Smoothing', [1 50]);
data = str2num(x{:});
handles.smt = data;
guidata(hObject,handles);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton5,'FontName','default')
clinical = handles.clinical;
controls_path = handles.controls_path;
mask = handles.mask;
smt = handles.smt;
in = 1;
resp = handles.resp;

nin = questdlg('Do you want to add a subject identifier (e.g., subject/scan number)?', ...
    '', ...
    'Yes','No',[1 50]);

switch nin
    case 'Yes'
        xx = inputdlg('Type name/number here',...
            '', [1 50]);
        
        handles.resp2 = nin;
        guidata(hObject,handles);
        
        handles.xx = xx;
        guidata(hObject,handles);
        
    case 'No'
        xx = {'NoID'};
        handles.resp2 = nin;
        guidata(hObject,handles);
        
        handles.xx = xx;
        guidata(hObject,handles);
        
end

resp2 = handles.resp2;
xx = handles.xx;

nin = questdlg('Do you want to use parallel computing?', ...
    '', ...
    'Yes','No',[1 50]);

switch nin
    case 'Yes'
        subject_level_MRI_stats_run_par
        
    case 'No'
        subject_level_MRI_stats_run
        
end

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton6,'FontName','default')
in_file = uigetfile('*.nii','Select z_map*.nii file for threholding (Random Field Theory)');
x = inputdlg({'Voxel p-threshold (e.g., 0.001)','Cluster p-threshold (e.g., 0.05)'},...
    'Threshold ALC-maps', [1 70; 1 70]);
VoxelPThreshold = str2num(x{1});
ClusterPThreshold = str2num(x{2});
mask = handles.mask;
xx = handles.xx;
Thresholding_RFT;

% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton7,'FontName','default')
in_file = uigetfile('*.nii','Select z_map*.nii file for threholding (False Discovery Rate)');
x = inputdlg({'Q-threshold (e.g., 0.05)'},...
    'Threshold ALC-maps', [1 70]);
QThreshold = str2num(x{1});
mask = handles.mask;
xx = handles.xx;
Thresholding_FDR;
