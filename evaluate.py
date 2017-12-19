#!/usr/bin/env python3

"""Script for evaluating HNMT models using WMT data.

This should be run on taito-gpu.csc.fi. The translation is launched on a
GPU node and should take 20-25 minutes (or longer if there is a queue).

For users without access to CSC, see the comments below for how to modify
this script.

Example:

python3 evaluate.py partial-22500 fi-en.model.22500 \
        wmt16/dev/newstest2015-fien-src.fi.sgm \
        wmt16/dev/newstest2015-fien-ref.en.sgm

A directory called `results` will be created, where translated files and
BLEU/NIST scores are written.
"""

import sys
import glob
from subprocess import Popen, PIPE, call
import stat
import os
import re


def main():
    use_gpu = False
    ident = sys.argv[1]
    model = ','.join(glob.glob(sys.argv[2]))
    xml_src = sys.argv[3]
    xml_trg_ref = sys.argv[4]
    extra_args = sys.argv[5:]
    assert xml_src.endswith('.sgm') and xml_src[-11:-6] == '-src.'
    src = xml_src[-6:-4]
    assert xml_trg_ref.endswith('.sgm') and xml_trg_ref[-11:-6] == '-ref.'
    trg = xml_trg_ref[-6:-4]

    detokenize = False
    source_tokenizer = 'word'

    script_dir = 'moses-scripts'
    scratch_dir = 'results'

    def run_perl(script, args=[], infile=None, outfile=None):
        stdin = None
        stdout = None
        if infile is not None: stdin = open(infile, 'rb')
        if outfile is not None: stdout = open(outfile, 'wb')
        r = call(['perl', os.path.join(script_dir, script)] + list(args),
                 stdin=stdin, stdout=stdout)
        if stdin: stdin.close()
        if stdout: stdout.close()
        return r

    def strip_xml(infile, outfile):
        cmd = [os.path.join(script_dir, 'strip-xml.perl')]
        with open(infile, 'rb') as inf:
            with Popen(cmd, stdin=inf, stdout=PIPE) as p:
                data = p.stdout.read()
        with open(outfile, 'wb') as outf:
            outf.write(re.sub(b'\n{2,}', b'\n', data.lstrip(b'\n')))

    def wrap_xml(infile, outfile, xml_src):
        run_perl('wrap-xml.perl', [trg, xml_src, 'HNMT'],
                 infile=infile, outfile=outfile)

    if not os.path.isdir(scratch_dir): os.mkdir(scratch_dir)

    ext = '.%s.sgm' % src
    assert xml_src.endswith(ext), (xml_src, ext)
    base = os.path.join(scratch_dir,
            ident + '-' + os.path.basename(xml_src)[:-len(ext)])
    raw_src = '%s.%s' % (base, src)
    raw_trg = '%s.%s' % (base, trg)
    xml_trg = '%s.%s.sgm' % (base, trg)
    if not os.path.exists(raw_src):
        strip_xml(xml_src, raw_src)
    else:
        print('Raw source text already available', file=sys.stderr)

    command = [
            'hnmt.py', '--load-model', model, '--translate', raw_src,
            '--output', raw_trg, '--beam-size', '10', '--source-tokenizer',
            source_tokenizer] + extra_args

    # Translate the source file
    # NOTE: replace this with whatever your system requires for launching
    #       GPU jobs
    # -------------------------------------------------------------------
    if use_gpu and not os.path.exists(raw_trg):
        slurm = os.path.join(base+'.sh')
        with open(slurm, 'w', encoding='utf-8') as f:
            f.write(r'''#!/bin/bash -l

module purge
module load python-env/3.4.1
module load cuda/8.0

THEANO_FLAGS=optimizer=fast_run,device=gpu,floatX=float32 python3 %s
''' % ' '.join(command))

        os.chmod(slurm, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        call(['srun', '-N', '1', '--gres=gpu:1', '--mem=8192', '-p', 'gpu',
              '-t', '01:00:00', slurm])
        os.remove(slurm)
    # -------------------------------------------------------------------
    elif not use_gpu and not os.path.exists(raw_trg):
        call(['python3'] + command)
    else:
        print('Translated text already available', file=sys.stderr)

    if detokenize:
        if not os.path.exists(raw_trg+'.detok'):
            run_perl('detokenizer.perl',
                     infile=raw_trg, outfile=raw_trg+'.detok')
        else:
            print('Detokenized text already available', file=sys.stderr)
        raw_trg = raw_trg+'.detok'
    if not os.path.exists(xml_trg):
        wrap_xml(raw_trg, xml_trg, xml_src)
    else:
        print('XML-wrapped text already available', file=sys.stderr)
    if not os.path.exists('%s.report'%base):
        run_perl('mteval-v13a.pl',
                 ['-s', xml_src, '-r', xml_trg_ref, '-t', xml_trg],
                 outfile='%s.report'%base)
    else:
        print('Report already available', file=sys.stderr)

if __name__ == '__main__': main()

