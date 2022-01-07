import os
import shutil
import subprocess

def create_girg_(n, d, ple, alpha, deg, wseed, pseed, sseed, threads, file, dot, edge):
	"""run script in build file"""
	os.system("./girgs/build/gengirg.app/Contents/MacOS/gengirg " + "-n " + str(n) + " -d " + str(d) + " -ple " + str(ple) + " -alpha " + str(alpha) + " -deg " + str(deg) + " -wseed " + str(wseed) + " -pseed " + str(pseed) + " -sseed " + str(sseed) + " -threads " + str(threads) + " -file " + str(file) + " -dot " + str(dot) + " -edge " + str(edge))
	os.rename(str(file) + ".txt", "./girgs/build/files/" + str(file) + ".txt")


def get_input(inputFile: str):
	shutil.copy2("./girgs/build/files/" + inputFile, "./input/")
