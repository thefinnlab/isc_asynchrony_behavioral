import sys
import os
import glob
import json

def attempt_makedirs(d):

	if not os.path.exists(d):
		try:
			os.makedirs(d)
		except Exception:
			pass