import os
import torch

def cp_projects(to_path):

    with open('./.gitignore','r') as fp:
        ign = fp.read()
    ign += '\n.git'
    spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
    all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files}
    matches = spec.match_files(all_files)
    matches = set(matches)
    to_cp_files = all_files - matches

    for f in to_cp_files:
        dirs = os.path.join(to_path,'code',os.path.split(f[2:])[0])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))

def parse_system(args):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_%s_%s' % (args.model, args.dataset)

    save_path = os.path.join(args.log_path, now + hyper_param_str + '_' + args.note)

    logger = DistSummaryWriter(save_path)

    config_txt = os.path.join(save_path, 'args')

    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(args))
        cp_projects(save_path)

    return logger, save_path
