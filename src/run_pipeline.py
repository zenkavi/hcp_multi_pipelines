#!/opt/miniconda-latest/envs/neuro/bin/python
# This script can be used to run fsl and spm pipelines with specific parameters. 
# Use : python run_pipeline.py -e /srv/tempdd/egermani/hcp_pipelines/data/original -r /srv/tempdd/egermani/hcp_pipelines/data/derived -s '["100206"]' -o '["l1"]' -S 'SPM' -t '["MOTOR"]' -c '["rh"]' -f 8 -p 0 -h 'derivatives'

from os.path import join as opj
import sys
import getopt
import json
import importlib
# import subprocess
import sys

# Install a single package
# subprocess.check_call([sys.executable, "-m", "pip", "install", "niflow-nipype1-workflows"])

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=RuntimeWarning)

if __name__ == "__main__":
    # Options
    subject_list = []
    exp_dir = None
    result_dir = None
    software = None
    operation = None
    task_list = []
    contrast_list = []
    fwhm_list=[]
    nb_param=[]
    hrf=[]

    try:
        OPTIONS, REMAINDER = getopt.getopt(sys.argv[1:], 'e:r:s:o:S:t:c:f:p:h:', ['exp_dir=', 'result_dir=', 'subjects=', 'operation=',
            'software=', 'task=', 'contrast=', 'fwhm=', 'param=', 'hrf='])

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # Replace variables depending on options
    for opt, arg in OPTIONS:
        if opt in ('-e', '--exp_dir'):
            exp_dir= arg
        elif opt in ('-r', '--result_dir'):
            result_dir = arg
        elif opt in ('-s', '--subjects'):
            subject_list = json.loads(arg)
        elif opt in ('-o', '--operation'): 
            operation = json.loads(arg) # preprocess, l1, registration
        elif opt in ('-S', '--software'):
            software = str(arg)
        elif opt in ('-t', '--task'):
            task_list = json.loads(arg)
        elif opt in ('-c', '--contrast'):
            contrast_list = json.loads(arg)
        elif opt in ('-f', '--fwhm'):
            fwhm_list.append(int(arg))
        elif opt in ('-p', '--param'):
            nb_param.append(int(arg))
        elif opt in ('-h', '--hrf'):
            hrf.append(str(arg))

    print('OPTIONS   :', OPTIONS)

    if len(operation) == 0:
        print('All operations will be performed.')
        operation = ['preprocessing', 'l1']

    # Load corresponding package
    package = 'lib.' + software + '_pipelines'
    pipeline = importlib.import_module(package)

    # Directories
    working_dir = f'{software}_preprocessing/intermediate_results'
    output_dir = f'{software}_preprocessing'

    if 'preprocessing' in operation:
        preprocess = pipeline.get_preprocessing(exp_dir, result_dir, working_dir, output_dir, subject_list, task_list, 
                                                fwhm_list)
        preprocess.run('MultiProc', plugin_args={'n_procs': 16})

    if 'l1' in operation:
        l1_analysis = pipeline.get_l1_analysis(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, 
                                                    contrast_list, fwhm_list, nb_param, hrf)
        l1_analysis.run('MultiProc', plugin_args={'n_procs':16})

    if 'registration' in operation:
        registration = pipeline.get_registration(exp_dir, output_dir, working_dir, result_dir, subject_list, task_list, 
            contrast_list, fwhm_list, nb_param, hrf)

        registration.run('MultiProc', plugin_args={'n_procs': 16})


