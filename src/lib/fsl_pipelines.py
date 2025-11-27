#python3
#This script contains function to create NiPype workflow for analysing HCP dataset with FSL.

from nipype.interfaces.fsl import (BET, FAST, MCFLIRT, FLIRT, FNIRT, ApplyWarp, SUSAN, 
								   Info, ImageMaths, ImageStats, Threshold, Level1Design, FEATModel, 
								   L2Model, Merge, FLAMEO, ContrastMgr,Cluster,  FILMGLS, Randomise, 
								   MultipleRegressDesign, ExtractROI, PlotMotionParams)
from nipype.algorithms.modelgen import SpecifyModel

from niflow.nipype1.workflows.fmri.fsl import create_reg_workflow, create_featreg_preproc
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.misc import Gunzip
from nipype import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.base import Bunch

from os.path import join as opj
import os

def get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, task_list, fwhm_list):
	"""
	Returns the FSL preprocessing workflow.
	Parameters: 
		- exp_dir: str, directory where raw data are stored
		- result_dir: str, directory where results will be stored
		- working_dir: str, name of the sub-directory for intermediate results
		- output_dir: str, name of the sub-directory for final results
		- subject_list: list of str, list of subject for which you want to do the preprocessing
		- task_list: list of str, list of task for which you want to do the preprocessing
		- fwhm_list: list of int, list of fhwm kernel for smoothing 
		
	Returns: 
		- preprocessing: Nipype WorkFlow 
	"""
	infosource_preproc = Node(IdentityInterface(fields = ['subject_id', 'task', 'fwhm']), 
		name = 'infosource_preproc')

	infosource_preproc.iterables = [('subject_id', subject_list), ('task', task_list), ('fwhm', fwhm_list)]

	# Templates to select files node
	anat_file = opj('STRUCTURAL', '{subject_id}', 'unprocessed', '3T', 'T1w_MPR1', 
					'{subject_id}_3T_T1w_MPR1.nii.gz')

	func_file = opj('{task}', '{subject_id}', 'unprocessed', '3T', 'tfMRI_{task}_LR', 
					'{subject_id}_3T_tfMRI_{task}_LR.nii.gz')

	template = {'anat' : anat_file, 'func' : func_file}

	# SelectFiles node - to select necessary files
	selectfiles_preproc = Node(SelectFiles(template, base_directory=exp_dir), name = 'selectfiles_preproc')
	
	img2float = Node(
		ImageMaths(
			out_data_type='float', op_string='', suffix='_dtype'),
		name='img2float')
	
	extract_mean = Node(ExtractROI(t_min = 142, t_size=1), name = 'extract_first')
	
	reg = create_reg_workflow()
	reg.inputs.inputspec.target_image = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
	reg.inputs.inputspec.target_image_brain = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
	
	mc_smooth = create_featreg_preproc(name='featpreproc',
						   highpass=True,
						   whichvol='middle',
						   whichrun=0)
	
	#mc_smooth.inputs.inputspec.fwhm = 5
	mc_smooth.inputs.inputspec.highpass = 100

	datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

	preprocessing =  Workflow(base_dir = opj(result_dir, working_dir), name = "preprocess_fsl")

	preprocessing.connect([(infosource_preproc, selectfiles_preproc, [('subject_id', 'subject_id'),
																		('task', 'task')]),
							(selectfiles_preproc, img2float, [('func', 'in_file')]),
							(img2float, extract_mean, [('out_file', 'in_file')]),
							(extract_mean, reg, [('roi_file', 'inputspec.mean_image')]),
							(selectfiles_preproc, reg, [('func', 'inputspec.source_files'), 
													   ('anat', 'inputspec.anatomical_image')]),
							(infosource_preproc, mc_smooth, [('fwhm', 'inputspec.fwhm')]),
							(selectfiles_preproc, mc_smooth, [('func', 'inputspec.func')]), 
							(reg, datasink, [('outputspec.anat2target_transform', 'preprocess_fsl.@transfo_all'), 
											('outputspec.func2anat_transform', 'preprocess_fsl.@transfo_init')]), 
							(mc_smooth, datasink, [('outputspec.motion_parameters', 
													'preprocess_fsl.@parameters_file'),
												  ('outputspec.highpassed_files', 'preprocess_fsl.@hp'),
												  ('outputspec.smoothed_files', 'preprocess_fsl.@smooth'), 
												  ('outputspec.mask', 'preprocess_fsl.@mask')])
							])

	return preprocessing



def get_subject_infos(event_file, contrasts):
	'''
	Create Bunchs for specifyModel.
	Parameters :
	- event_file: list of str, event files for the subject
	- contrasts: list of contrast to analyze
	
	Returns :
	- subject_info : Bunch for 1st level analysis.
	'''
	from nipype.interfaces.base import Bunch
	import numpy as np
	
	# Global lists 
	cond_names = sorted(contrasts) 
	onsets = []
	durations = []

	# Selection of only event files corresponding to selected contrasts
	event_files = [f for f in sorted(event_file) if f.split('/')[-1].split('.')[0] in contrasts]

	for i, c in enumerate(sorted(contrasts)):
		onset = [] # For each contrast, save the onset and duration
		duration = []
		file = sorted(event_files)[i]
		with open(file, 'rt') as f:
			for line in f:
				info = line.strip().split()

				onset.append(float(info[0])) 
				duration.append(float(info[1]))
		onsets.append(onset)
		durations.append(duration) # Add these onset and duration to global list 
					
	subject_info = Bunch(conditions=cond_names,
							 onsets=onsets, # Onsets and Durations contain list of lists with onsets and durations for every contrast
							 durations=durations,
							 amplitudes=None,
							 regressor_names=None,
							 regressors=None)

	return subject_info

def get_24_param(param_file):
    # Function to apply bash script on parameter file to obtain 24 MC parameters from the 6 firsts
    import os 

    out_file = os.path.join(os.path.dirname(param_file), '24_' + os.path.basename(param_file))[:-10] + 'txt'

    os.system(f'bash /srv/tempdd/egermani/hcp_pipelines/src/lib/mp_diffpow24.sh {param_file} {out_file}')

    return out_file

def get_l1_analysis(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, contrast_list, fwhm_list, nb_param, hrf):
	"""
	Returns the FSL first level analysis workflow.
	Parameters: 
		- exp_dir: str, directory where raw data are stored
		- result_dir: str, directory where results will be stored
		- working_dir: str, name of the sub-directory for intermediate results
		- output_dir: str, name of the sub-directory for final results
		- subject_list: list of str, list of subject for which you want to do the analysis
		- task_list: list of str, list of task for which you want to do the analysis
		- contrast_list: list of str, list of contrast for which you want to do the analysis
		- fwhm_list: list of int, list of fwhm kernel for smoothing 
		- nb_param: list of int, list of number of parameters for model -> one of 0, 6 or 24
		- hrf: list of string, derivatives for hrf function (derivatives or no_derivatives)

	Returns: 
		- l1_analysis : Nipype WorkFlow 
	"""
	# Infosource Node - To iterate on subjects
	infosource = Node(IdentityInterface(fields = ['subject_id', 'task', 'fwhm', 'nb_param', 'hrf']),
					  name = 'infosource')

	infosource.iterables = [('subject_id', subject_list), ('task', task_list), ('fwhm', fwhm_list), ('nb_param', nb_param), ('hrf', hrf)]

	# Templates to select files node
	param_file = opj(output_dir, 'preprocess_fsl', '_fwhm_{fwhm}_subject_id_{subject_id}_task_{task}', 
	 '_realign0', '{subject_id}_3T_tfMRI_{task}_LR_dtype_mcf.nii.gz.par')

	func_file = opj(output_dir, 'preprocess_fsl', '_fwhm_{fwhm}_subject_id_{subject_id}_task_{task}', '_addmean0',
	'{subject_id}_3T_tfMRI_{task}_LR_dtype_mcf_mask_smooth_mask_gms_tempfilt_maths.nii.gz')

	event_file = opj(exp_dir, '{task}', '{subject_id}', 'unprocessed', '3T', 'tfMRI_{task}_LR', 'LINKED_DATA', 'EPRIME', 'EVs', '*.txt')

	template = {'param' : param_file, 'func' : func_file, 'event' : event_file}

	# SelectFiles node - to select necessary files
	selectfiles = Node(SelectFiles(template, base_directory=result_dir), name = 'selectfiles')

	# DataSink Node - store the wanted results in the wanted repository
	datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

	# Get Subject Info - get subject specific condition information
	subject_infos = Node(Function(input_names=['event_file', 'contrasts'],
								   output_names=['subject_info'],
								   function=get_subject_infos),
						  name='subject_infos')

	subject_infos.inputs.contrasts = contrast_list

	specify_model = Node(SpecifyModel(high_pass_filter_cutoff = 60,
									 input_units = 'secs',
									 time_repetition = 0.72), name = 'specify_model')

	if hrf == ['derivatives']:
		hrf_values = True
		print('Use derivatives')
	else:
		hrf_values = False
		print('No use derivatives')

	l1_design = Node(Level1Design(bases = {'dgamma':{'derivs' : hrf_values}},
								 interscan_interval = 0.72, 
								 model_serial_correlations = True), name = 'l1_design')

	l1_design.inputs.contrasts = [(f'{c} vs baseline', 'T', [c], [1]) for c in sorted(contrast_list)]

	model_generation = Node(FEATModel(), name = 'model_generation')

	model_estimate = Node(FILMGLS(full_data=False, autocorr_noestimate=True), name='model_estimate')

	# Create l1 analysis workflow and connect its nodes
	l1_analysis = Workflow(base_dir = opj(result_dir, working_dir), name = "l1_analysis_fsl")

	l1_analysis.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
												   ('task', 'task'), ('fwhm', 'fwhm')]),
						(selectfiles, subject_infos, [('event', 'event_file')]),
						(selectfiles, specify_model, [('func', 'functional_runs')]),
						(subject_infos, specify_model, [('subject_info', 'subject_info')]),
						(specify_model, l1_design, [('session_info', 'session_info')]),
						(l1_design, model_generation, [('ev_files', 'ev_files'), ('fsf_files', 'fsf_file')]),
						(selectfiles, model_estimate, [('func', 'in_file')]),
						(model_generation, model_estimate, [('con_file', 'tcon_file'), 
															('design_file', 'design_file')]),
						(model_estimate, datasink, [('results_dir', 'l1_analysis_fsl.@results')]),
						(model_generation, datasink, [('design_file', 'l1_analysis_fsl.@design_file'),
													 ('design_image', 'l1_analysis_fsl.@design_img')]),
						])

	if nb_param == [6]:
		print('Use 6 MC param')
		l1_analysis.connect([(selectfiles, specify_model, [('param', 'realignment_parameters')])])
	elif nb_param == [24]:
		print('Use 24 MC param')
		param_extent = Node(Function(input_names=['param_file'],
	   output_names=['out_file'],
	   function=get_24_param),
	  name='param_extent')

		l1_analysis.connect([(selectfiles, param_extent, [('param', 'param_file')]), 
		(param_extent, specify_model, [('out_file', 'realignment_parameters')])])

	return l1_analysis

def get_registration(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, contrast_list, fwhm_list, param_list, hrf):
	"""
	Returns the FSL registration workflow to apply AFTER L1 analysis.
	Parameters: 
		- exp_dir: str, directory where raw data are stored
		- result_dir: str, directory where results will be stored
		- working_dir: str, name of the sub-directory for intermediate results
		- output_dir: str, name of the sub-directory for final results
		- subject_list: list of str, list of subject for which you want to do the analysis
		- task_list: list of str, list of task for which you want to do the analysis
		- contrast_list: list of str, list of contrast for which you want to do the analysis
		- fwhm_list: list of int, list of fwhm kernel for smoothing 
		- param_list: list of int, list of number of parameters for model -> one of 0, 6 or 24
		- hrf: list of string, derivatives for hrf function (derivatives or no_derivatives)

	Returns: 
		- registration : Nipype WorkFlow 
	"""
	# Infosource Node - To iterate on subjects
	infosource = Node(IdentityInterface(fields = ['subject_id', 'task', 'contrast', 'fwhm', 'nb_param', 'hrf']),
					  name = 'infosource')

	num_contrast = [i for i in range(1, len(contrast_list)+1)]

	infosource.iterables = [('subject_id', subject_list), ('task', task_list), 
	('contrast', num_contrast), ('fwhm', fwhm_list), ('nb_param', param_list), ('hrf', hrf)]

	func2anat_transform_file = opj(output_dir, 'preprocess_fsl', '_fwhm_{fwhm}_subject_id_{subject_id}_task_{task}', 
		'{subject_id}_3T_tfMRI_{task}_LR_dtype_roi_flirt.mat')
	anat2target_transform_file = opj(output_dir, 'preprocess_fsl', '_fwhm_{fwhm}_subject_id_{subject_id}_task_{task}', 
		'{subject_id}_3T_T1w_MPR1_fieldwarp.nii.gz')
	cope_file = opj(output_dir, 'l1_analysis_fsl', '_fwhm_{fwhm}_hrf_{hrf}_nb_param_{nb_param}_subject_id_{subject_id}_task_{task}',
		'results', 'cope{contrast}.nii.gz')
	stat_file = opj(output_dir, 'l1_analysis_fsl', '_fwhm_{fwhm}_hrf_{hrf}_nb_param_{nb_param}_subject_id_{subject_id}_task_{task}',
		'results', '*stat{contrast}.nii.gz')

	template = {'anat2target_transform': anat2target_transform_file, 
					'func2anat_transform': func2anat_transform_file, 
					'cope': cope_file, 'stat':stat_file}

	# SelectFiles node - to select necessary files
	selectfiles = Node(SelectFiles(template, base_directory=result_dir), name = 'selectfiles')

	# DataSink Node - store the wanted results in the wanted repository
	datasink = Node(DataSink(base_directory=result_dir, container=output_dir), name='datasink')

	warpall_cope = MapNode(ApplyWarp(interp='spline'),name='warpall_cope', iterfield=['in_file'])
	warpall_cope.inputs.ref_file = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
	warpall_cope.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

	warpall_stat = MapNode(ApplyWarp(interp='spline'),name='warpall_stat', iterfield=['in_file'])
	warpall_stat.inputs.ref_file = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
	warpall_stat.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

	# Create registration workflow and connect its nodes
	registration = Workflow(base_dir = opj(result_dir, working_dir), name = "registration_fsl")

	registration.connect([(infosource, selectfiles, [('subject_id', 'subject_id'),
												   ('task', 'task'), ('fwhm', 'fwhm'), 
												   ('contrast', 'contrast'), 
												   ('nb_param', 'nb_param'), ('hrf', 'hrf')]),
						(selectfiles, warpall_cope, [('func2anat_transform', 'premat'), 
							('anat2target_transform', 'field_file'), 
							('cope', 'in_file')]), 
						(selectfiles, warpall_stat, [('func2anat_transform', 'premat'), 
							('anat2target_transform', 'field_file'), 
							('stat', 'in_file')]), 
						(warpall_cope, datasink, [('out_file', 'registration_fsl.@reg_map')]),
						(warpall_stat, datasink, [('out_file', 'registration_fsl.@reg_stat_map')])])

	return registration
