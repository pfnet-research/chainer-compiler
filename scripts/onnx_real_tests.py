import glob
import json
import os
import subprocess

from test_case import TestCase


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


class _Downloader(object):
    def __init__(self, json_content, json_data):
        self.json_content = json_content
        self.name = json_data['model_name']
        self.url = json_data['url']

    def test_name(self):
        return 'onnx_real_%s' % self.name

    def prepare(self):
        name = self.name
        url = self.url
        stamp = 'out/onnx_real_%s/data.json' % name
        if os.path.exists(stamp):
            with open(stamp) as f:
                if self.json_content == f.read():
                    return

        print('Downloading %s...' % url)
        subprocess.check_call('curl %s | tar -C out -xz' % url, shell=True)
        os.rename('out/%s' % name, 'out/%s' % self.test_name())
        with open(stamp, 'w') as f:
            f.write(self.json_content)


def get():
    tests = []
    for data_json in glob.glob('onnx/onnx/backend/test/data/real/*/data.json'):
        with open(data_json) as f:
            json_content = f.read()
            json_data = json.loads(json_content)

        downloader = _Downloader(json_content, json_data)
        rtol = None
        if downloader.name == 'densenet121':
            rtol = 1e-3
        # They need a large rtol probably due to LRN.
        if (downloader.name == 'bvlc_alexnet' or
            downloader.name == 'zfnet512'):
            rtol = 20
        tests.append(TestCase('out', downloader.test_name(),
                              prepare_func=downloader.prepare,
                              want_gpu=True,
                              rtol=rtol))
    return tests
