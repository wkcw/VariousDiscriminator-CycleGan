import math
import numpy as np

def compute_change(x, y):
    return np.abs(x-y)

import httplib, urllib, base64

headers = {
    # Request headers. Replace the placeholder key below with your subscription key.
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '217bc233c2864995bd51a1f3f9710aef',
}

params = urllib.urlencode({
})

# Replace the example URL below with the URL of the image you want to analyze.
body = "{ 'url': '' }"

try:
    # NOTE: You must use the same region in your REST call as you used to obtain your subscription keys.
    #   For example, if you obtained your subscription keys from westcentralus, replace "westus" in the
    #   URL below with "westcentralus".
    conn = httplib.HTTPSConnection('westus.api.cognitive.microsoft.com')
    conn.request("POST", "/emotion/v1.0/recognize?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

#a_diff = 0
#b_diff = 0
#for key, value in real_dict.iteritems():
#    print key
#    a_diff += compute_change(Gsoftmax4070_dict[key], real_dict[key])
#    b_diff += compute_change(rf70_dict[key], real_dict[key])
#print a_diff
#print b_diff