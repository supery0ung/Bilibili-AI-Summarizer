"""Bilibili WBI singing utility."""

import time
import hashlib
from functools import reduce

def get_mixin_key(ae):
    oe = [
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
        33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
        61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
        36, 20, 34, 44, 52
    ]
    le = reduce(lambda s, i: s + ae[i], oe, "")
    return le[:32]

def enc_wbi(params: dict, img_key: str, sub_key: str):
    mixin_key = get_mixin_key(img_key + sub_key)
    curr_time = int(time.time())
    params['wts'] = curr_time
    params = dict(sorted(params.items()))
    # 过滤 value 中的非法字符
    params = {
        k: ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
        for k, v in params.items()
    }
    from urllib.parse import urlencode
    query = urlencode(params)
    w_rid = hashlib.md5((query + mixin_key).encode('utf-8')).hexdigest()
    params['w_rid'] = w_rid
    return params
