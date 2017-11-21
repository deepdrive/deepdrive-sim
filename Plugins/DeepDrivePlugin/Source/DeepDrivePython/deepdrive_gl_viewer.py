#!

# This is statement is required by the build system to query build info

#
# Ported to PyOpenGL 2.0 by Tarn Weisner Burton 10May2001
#
# This code was created by Richard Campbell '99 (ported to Python/PyOpenGL by John Ferguson 2000)
#
# The port was based on the PyOpenGL tutorial module: dots.py  
#
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import ctypes

import numpy

import deepdrive
import platform

# Number of the glut window.
window = 0
program = 0
vao = 0
textures = None
depth_textures = None

location_offset = 0
location_size = 0
location_gamma = 0

VERT_SOURCE = """
#version 330
in vec4 aVertex;
out vec2 vTexCoord;
uniform vec2 uOffset;
uniform vec2 uSize;
void main()
{
	gl_Position = vec4(uOffset + aVertex.xy * uSize, 0.0, 1.0);
	vTexCoord = aVertex.zw;
}"""
FRAG_SOURCE = """
#version 330
uniform sampler2D colorMap;
uniform float uGamma;
in vec2 vTexCoord;
out vec4 frag_color;
void main()
{
	vec3 color = texture2D(colorMap, vTexCoord).rgb;
    frag_color = vec4( pow(color, vec3(uGamma)), 1.0 );
}"""


VERTICES =	[	0.0, 0.0, 0.0, 1.0
			,	1.0, 0.0, 1.0, 1.0
			,	0.0, 1.0, 0.0, 0.0
			,	1.0, 1.0, 1.0, 0.0
			]
VERTICES = numpy.array(VERTICES, dtype=numpy.float32)

def LoadTexture(index, width, height, data):
	glBindTexture(GL_TEXTURE_2D, textures[index])
	#glTexImage2D(GL_TEXTURE_2D, 0, GL_HALF_FLOAT, width, height, 0, GL_RGB, GL_HALF_FLOAT, data)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, data.astype(numpy.float32))

def LoadDepthTexture(index, width, height, data):
	global depth_textures
	glBindTexture(GL_TEXTURE_2D, depth_textures[index])
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_FLOAT, data.astype(numpy.float32))


def CreateShader(src, type):
	shader = glCreateShader(type)
	glShaderSource(shader, src)
	glCompileShader(shader)
	return shader

def InitGL(Width, Height):				# We call this right after our OpenGL window is created.
	global program
	global vao
	global textures
	global depth_textures
	global location_offset
	global location_size
	global location_gamma
	glClearColor(0.0, 1.0, 0.0, 0.0)
	glDepthFunc(GL_LESS)
	glDisable(GL_DEPTH_TEST)
	glDisable(GL_CULL_FACE)

	#	setup vertex shader
	vtxShader = CreateShader(VERT_SOURCE, GL_VERTEX_SHADER)
	if not glGetShaderiv(vtxShader, GL_COMPILE_STATUS):
		sys.stderr.write("Error: Could not compile vertex shader.\n")
		sys.stderr.write("Error: {0}\n".format(glGetShaderInfoLog(vtxShader)))
		exit(2)

	#	setup fragment shader
	frgShader = CreateShader(FRAG_SOURCE, GL_FRAGMENT_SHADER)
	if not glGetShaderiv(frgShader, GL_COMPILE_STATUS):
		sys.stderr.write("Error: Could not compile fragment shader.\n")
		sys.stderr.write("Error: {0}\n".format(glGetShaderInfoLog(frgShader)))
		exit(2)
	print('Shader successfully compiled')

	#	link program
	program = glCreateProgram()
	glAttachShader(program, vtxShader)
	glAttachShader(program, frgShader)
	glLinkProgram(program)
	if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
		sys.stderr.write("Error: {0}\n".format(glGetProgramInfoLog(program)))
		exit(4)

	glUseProgram(program)
	location = glGetUniformLocation(program, "colorMap")
	if location >= 0:
		glUniform1i(location, 0)

	location_offset = glGetUniformLocation(program, "uOffset")
	location_size = glGetUniformLocation(program, "uSize")
	location_gamma = glGetUniformLocation(program, "uGamma")

	print('Program ready', location_offset, location_size)

	vao = glGenVertexArrays(1)
	glBindVertexArray(vao)
	vertex_buffer = glGenBuffers(1)
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
	glBufferData(GL_ARRAY_BUFFER, len(VERTICES) * 4, VERTICES, GL_STATIC_DRAW)    
	glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, ctypes.c_void_p(0))
	glBindBuffer(GL_ARRAY_BUFFER, 0)
	glBindVertexArray(0)

	textures = glGenTextures(4)
	for texId in textures:
		glBindTexture(GL_TEXTURE_2D, texId)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

	depth_textures = glGenTextures(4)
	for texId in depth_textures:
		glBindTexture(GL_TEXTURE_2D, texId)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	print('GLError:', glGetError())


def Resize(Width, Height):
	if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
		Height = 1

	glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation

def DrawImage(offset, size, textureId):
	global location_offset
	global location_size
	global location_gamma
	glBindTexture(GL_TEXTURE_2D, textureId)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
	glBindVertexArray(vao)
	glEnableVertexAttribArray(0)
	glUniform2f(location_size, size[0], size[1])
	glUniform2f(location_offset, offset[0], offset[1])
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)


def Render():
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glUseProgram(program)
	glActiveTexture(GL_TEXTURE0)

	glUniform1f(location_gamma, 0.45)
	DrawImage((-0.9, 0.2), (0.6, 0.6), textures[0])
	DrawImage((0.1, 0.2), (0.6, 0.6), textures[1])

	glUniform1f(location_gamma, 1.0)
	DrawImage((-0.9, -0.8), (0.6, 0.6), depth_textures[0])
	DrawImage((0.1, -0.8), (0.6, 0.6), depth_textures[1])

	glutSwapBuffers()

def Idle():
	snapshot = deepdrive.step()
	if snapshot:
		print(snapshot.sequence_number, snapshot.speed, snapshot.is_game_driving, snapshot.camera_count)
		print('Position    :', snapshot.position)
		print('Rotation    :', snapshot.rotation)
		print('Velocity    :', snapshot.velocity)
		print('Acceleration:', snapshot.acceleration)
		print('Dimension   :', snapshot.dimension)

		ind = 0
		for cc in snapshot.cameras:
			if ind < 4:
				LoadTexture(ind, cc.capture_width, cc.capture_height, cc.image_data)
				LoadDepthTexture(ind, cc.capture_width, cc.capture_height, cc.depth_data)
			ind = ind + 1
			print('  Camera:', cc.type, cc.id, cc.capture_width, 'x', cc.capture_height)
			print('    image size', len(cc.image_data), cc.image_data[0], cc.image_data[1], cc.image_data[2])
			print('    depth size', len(cc.depth_data), cc.depth_data[0], cc.depth_data[1], cc.depth_data[2])


		glutPostRedisplay()

def onKeyPressed(*args):
	global window

	if args[0] == b'\x1b':
		glutLeaveMainLoop()

def main():
	global window

	if platform.system() == 'Linux':
		connected = deepdrive.reset('/tmp/deepdrive_shared_memory', 157286400)
	elif platform.system() == 'Windows':
		connected = deepdrive.reset('Local\DeepDriveCapture', 157286400)

	if connected:

		glutInit(sys.argv)

		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
		glutInitWindowSize(1200, 800)
		glutInitWindowPosition(0, 0)
		
		window = glutCreateWindow(b'DeepDrive viewer')
		glutDisplayFunc(Render)
		
		# Uncomment this line to get full screen.
		#glutFullScreen()

		glutIdleFunc(Idle)
		
		glutReshapeFunc(Resize)
		glutKeyboardFunc(onKeyPressed)

		InitGL(640, 480)
		glutMainLoop()


print ('Hit ESC key to quit.')
main()
