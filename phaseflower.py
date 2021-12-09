#-------------------------------------------------------------------------------
# Name:        PhaseFlowEr
# Purpose:
#
# Author:      toddh
#
# Created:     12/08/2015
# Copyright:   (c) toddh 2015

# this version keeps a 2-D array of complex numbers
# which get rotated w/ multiply instead of a sin()
# might be faster might not be

#-------------------------------------------------------------------------------

import datetime

import os

import math
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from skimage import color
from skimage.filters import gaussian as gaussian

import json
import uuid

# time parameters

msecs = 32
mfps = 30
mframes = msecs * mfps

mfcount = 0
def countframe():
    global mfcount
    mfcount += 1

# phase pharameters

pstart = 0
pstop =  -2 * math.pi     # negative looks positive
pstep = (pstop - pstart) / mframes

def setlength(seconds, fps=30, extraFrames=0):
    global msecs, mfps, mframes, pstop, pstep
    if seconds < 0.0:
        pstop =  2 * math.pi # I don't know why positive looks negative
        seconds = -seconds
    else:
        pstop = -2 * math.pi
    msecs = seconds
    mfps = fps
    mframes = max(1, round( msecs * mfps ))
    pstep = (pstop - pstart) / mframes
    mframes += extraFrames

# Some functions to change the default from float64 to whatever
from functools import partial

fdtype = np.float32

nparray = partial(np.array, dtype=fdtype)
npempty = partial(np.empty, dtype=fdtype)
npones  = partial(np.ones,  dtype=fdtype)
npzeros = partial(np.zeros, dtype=fdtype)
nplinspace = partial(np.linspace, dtype=fdtype)

def ranf():
    """make a random number that's perfectly
    accurate to six places"""
    return np.around(np.random.ranf(), 6)

"""
One tuple ta = (i,j,k,l) defaults
one tuple tb = (w,x,y,z) user input

precedence tb > ta

[ b if b != None else a for a,b in zip(ta,tb) ]
"""

def tuprek(ta,tb):
    """Replace items in ta with tb if tb item is not None"""
    return type(tb) ( [ b if b != None else a for a,b in zip(ta,tb) ] )

# size of the field in pixels and math

class zoomer:
    def set( xyz ):
        """Takes tuple of (width, height, zoom, center x, center y) any of
           which may be None.
        """
        #global xratio, yratio,  xsize, xstart, xstop, ysize, ystart, ystop, mdpi, mzoom
    #    y2x, x2y, xstep, ystep,

        deef = tuple( [0, 0, 1.0, 0.0, 0.0] )
        fxyz = tuprek(deef, xyz)
        x,y,z,cx,cy = fxyz

        if x == y == 0:
            x,y = 640, 360
        elif x == 0:
            x = y * 16 / 9
        elif y == 0:
            y = x * 9 / 16

        zoomer.xratio = 16.0
        zoomer.yratio = zoomer.xratio * y / x

        y2x = zoomer.xratio / zoomer.yratio
        x2y = 1.0 / y2x

        zoomer.zoom = max(z, 0.001)    # to prevent any divide-by-zero

        zoomer.cx = cx
        zoomer.cy = cy

        zoomer.xsize  =  int(x)
        zoomer.xstart = ( -2*math.pi / zoomer.zoom ) + cx
        zoomer.xstop  = (  2*math.pi / zoomer.zoom ) + cx
        #xstep  = (xstop - xstart) / xsize

        zoomer.ysize  =  int(zoomer.xsize * x2y)
        zoomer.ystart = (  2*math.pi * x2y / zoomer.zoom ) + cy
        zoomer.ystop  = ( -2*math.pi * x2y / zoomer.zoom ) + cy
        #ystep  = (ystop - ystart) / ysize

        zoomer.dpi = zoomer.xsize / 16

    def jsonable():
        return dict( x=zoomer.xsize, y=zoomer.ysize, z=zoomer.zoom,
                     cx=zoomer.cx, cy=zoomer.cy )

# A wavefield(freq, amp, cx, cy) is a single wave of frequency freq,
# amplitude amp, emanating from (cx,cy)

class wavefield:
    def __str__(self):
        return json.dumps( self.jsonable() )

    def jsonable(self):
        return {
            'freq' : self.freq,
            'amp' : self.amp,
            'phase' : self.phase,
            'exp' : self.exp,
            'cx' : self.cx,
            'cy' : self.cy
        }

    def __init_args__(self, freq=1.0, amp=1.0, phase=0.0, exp=1.0, cx=0.0, cy=0.0):
        self.amp = amp
        self.phase = phase
        self.exp = exp
        self.omega = freq * pstep
        self.t = 0.0

        self.freq = freq
        self.cx = cx
        self.cy = cy

        # make array with squares of distance from (cx,cy)

        xstart, xstop, xsize = zoomer.xstart, zoomer.xstop, zoomer.xsize
        ystart, ystop, ysize = zoomer.ystart, zoomer.ystop, zoomer.ysize

        lx = freq * (nplinspace(xstart, xstop, xsize) - cx)
        nx = npzeros( (ysize, xsize) )          # nx is a Y-first array
        nx[:,:] = lx                        # put an lx in each row

        ly = freq * (np.linspace(ystart, ystop, ysize) - cy)
        ny = npzeros( ( xsize, ysize ) )       # ny is _X-first_ so we can do...
        ny[:,:] = ly                            # copy ly into all rows
        ny = ny.swapaxes(0,1)                   # now Y-first all filled nice!

        zz = nx * nx + ny * ny

        self.z = np.sqrt(zz)        # now it's distance from (cx,cy) == Phase Plane
        self.z += phase             # advance by phase angle

        # vector plane setup

        cz = np.cos(self.z)
        sz = np.sin(self.z)
        self.vz = (cz + 1j * sz).astype(np.complex64)
        self.phasor = np.complex64( np.cos(self.omega) + np.sin(self.omega) * 1j )

        # continuting...

        self.calcframe()        # initializes self.s

    def __init_random__(self):
        fx = np.around( -3.0 * math.pi + 6.0 * math.pi * np.random.ranf(), 6 )
        fy = np.around( -3.0 * math.pi + 6.0 * math.pi * np.random.ranf(), 6 )

        # frequency is an integer for easy looping!  (Diff from FlowPhaser?)
        ff = np.random.randint(1,5) # np.random.choice( [1,2,4,8] )
        if np.random.choice( [True, False] ):
            ff = -ff

        # amplitude 0..1
#        fa = np.random.ranf() * 4.0 / abs(ff)
        fa = np.around( np.random.ranf(), 6 )

        # phase 0..2pi
        fp = np.around( 2.0 * math.pi * np.random.ranf(), 6 )

        # exponent - odd to preserve sign
        fe = random.choice([1,1,1,1,
                            3,3,3,
                            5,5,7])

        self.__init_args__(freq=ff, amp=fa, phase=fp, exp=fe, cx=fx, cy=fy)

    def __init__(self, **kwargs):
        if len(kwargs) == 0:        # empty dict
            self.__init_random__()
        else:
            self.__init_args__(**kwargs)

    # returns s (a Y-first ndarray)
    def gets(self):
        return self.rs

    def calcframe(self):
        #self.s = np.power(np.sin(self.z), self.exp) * self.amp # / 2.0 + 0.5    # -1,1 to 0,1
        self.t += self.omega
        #self.z += self.omega

        self.rs =  np.imag(self.vz) * self.amp
        self.vz *= self.phasor

# end class wavefield

from collections import OrderedDict

def SortedDict(dd):
    return OrderedDict( sorted(dd.items()) )

class fieldlist(list):
    guid = uuid.UUID( '0' * 32 )
    zoom = 1.0

    def __init_random__(self):
        self.guid = uuid.uuid4()
        super().__init__( [ wavefield() for i in range( random.randint(2,9) ) ] )

    def __init_data__(self, data):
        self.guid = data['guid']
        waves = data['waves']
        wavelist = [ wavefield( **wave ) for wave in waves ]
        super().__init__( wavelist )

    def __init__(self, data=None):
        if data == None:
            self.__init_random__()
        else:
            self.__init_data__(data)    # self.blah() can be in subclass!

    def __str__(self):
        """Pretty-print the upper structure,
           but keep lower parts on single lines.
           Was: return json.dumps(self.jsonable())"""
        IND = '    '    # indentation
        simp = self.jsonable() # it is named 'simp' because the test file was 'simple.json'
        ss  = '{\n'
        ss += IND + '"colorspace": ' + json.dumps(simp['colorspace']) + ',\n'
        ss += IND + '"coloroptions": ' + json.dumps( simp['coloroptions'] ) + ',\n'
        ss += IND + '"wavelist":\n'
        ss += IND + '{\n'
        ss += IND*2 + '"zoom": ' + json.dumps(SortedDict( simp['wavelist']['zoom'] )) + ',\n'
        ss += IND*2 + '"guid": ' + json.dumps(simp['wavelist']['guid']) + ',\n'
        ss += IND*2 + '"waves":\n'
        ss += IND*2 + '[\n'
        wavs = simp['wavelist']['waves']
        ss += IND*3 + (',\n' + IND*3).join([ json.dumps( SortedDict(e) ) for e in wavs ]) + '\n'
        ss += IND*2 + ']\n'
        ss += IND + '}\n'
        ss += '}'
        return ss

    def jsonable(self):
        waves = [ e.jsonable() for e in self ]
        z = zoomer.jsonable()
        return dict( guid=str(self.guid), waves=waves, zoom=z )

    def calcframe(self):
        for e in self:
            e.calcframe()

    def getframe(self): #this is not super() called by any extant subclass
        zp = np.dstack( [e.gets() for e in self] )          # now Y-first shape like (160,90,3)
        return zp   # 1 layer of pixel color [[[r,g,b]]] or [[[w,h,a,t,e,v,e,r]]]

# end class fieldlist

# Okay we start real simple then HSV it
# This class is a list of grayscale fields, no RGB, no HSV, just monochrome

class grayfieldlist(fieldlist):
    # returns average of all fields yes I mean mean you mean meanie
    def getframe(self):
        layerlist = [ e.gets() for e in self ]
        layers = nparray( layerlist )
        mean = layers.mean(0) / 2.0 + 0.5
        return mean

class coloroptions:
    def __init_data__(self, data):
        self.names = data[0]
        self.bands = data[1]
        self.invs  = data[2]

    def __init_random__(self, names):
        self.names = names
        self.bands = list()
        self.invs  = list()
        for i in range(len(names)):
            self.bands.append( random.choice( [1,2,4,8] ) )
            self.invs.append(  random.choice( [True, False] ) )

    def __init__(self, lonol):  # list of [ names ] or [ [list] [list] [list ] ]
        if   type(lonol[0]) == str:
            self.__init_random__(lonol)
        elif type(lonol[0]) == list:
            self.__init_data__(lonol)
        else:
            es = 'coloroptions constructor expects list of strs or lists, {} given.'
            raise ValueError( es.format( type(lonol[0]) ) )

    def jsonable(self):
        return [ self.names, self.bands, self.invs ]

    def process(self, grayframe):
        """For each color component, make a plane from
           grayframe that is multiplied by its number
           of bands then modulo chopped back between 0..1
           and possibly inverted.
           Return a np.array() of the chopped planes."""
        bandl = list()
        for i in range(len(self.names)):
            bf = ( (self.bands[i] * grayframe) % 1.0 )  # multiply by # bands and modulo chop so it's 0...1
            if self.invs[i]:
                bf = 1.0 - bf
            bandl.append(bf)
        rolled = np.dstack( bandl )
        return rolled

# Colorfields is a list of fields whose frame (from getframe()) is chopped
# according to its coloroptions():

class colorfields(grayfieldlist):
    options = None
    colorspace = None
    colornames = None

    def __init_random__(self, names):
        self.options = coloroptions( names )    # the options
        super().__init_random__()               # the wave fields

    def __init_data__(self, data):
#        print( json.dumps(data) )
        self.options = coloroptions( data['coloroptions'] )    # the options
        super().__init_data__(  data['wavelist'] )             # the wave fields

    def __init__(self, data=None):
        if data == None:
            self.__init_random__( self.colornames )
        else:
            self.__init_data__(data)

    def jsonable(self):
        wavel = super().jsonable()
        copts = self.options.jsonable()
        space = self.colorspace
        return dict(wavelist=wavel, coloroptions=copts, colorspace=space)

    # returns np.array( [[[c1,c2,c3]]] )
    def getframe(self):
        grayframe = super().getframe()
        ccc = self.options.process(grayframe)
        return ccc

    def cycle(self, writer, img):
        start = datetime.datetime.now()
        opie = "{:3}%  frame {:5} of {:5} time {}"
        ends = { True : '\n', False : '\r'}
        for phrame in range(0, mframes):

            percent =  int( round( phrame / mframes, 2 ) * 100.0 )
            elapsed = str(datetime.datetime.now() - start)
            print(opie.format(percent, phrame, mframes, elapsed[:-4]), end=ends[phrame == mframes])

            rgb = self.getframe() # self. will try *subclass* getframe first
#            gauss = gaussian( rgb, 1.5, multichannel=True )  # set multichannel so it doesn't warn about guessing
            img.set_data( rgb ) # gauss )
            writer.grab_frame()
            countframe()
            self.calcframe()

# end class Colorfields

# This one interprets its data as RGB - no translation necessary, just names
class rgbfields(colorfields):
    colorspace = 'rgb'
    colornames = ['red', 'green', 'blue']

# This one interprets its data as HSV
class hsvfields(colorfields):
    colorspace = 'hsv'
    colornames = ['hue', 'sat', 'val']

    def getframe(self):
        hsv = super().getframe()
        rgb = color.hsv2rgb(hsv)
        return rgb

# This one interprets its data as XYZ
class xyzfields(colorfields):
    colorspace = 'xyz'
    colornames = ['x', 'y', 'z']

    def getframe(self):
        xyz = super().getframe()
        rgb = color.xyz2rgb(xyz)
        return rgb

# This interpets itself as L*u*v
class luvfields(colorfields):
    colorspace = 'luv'
    colornames = ['l', 'u', 'v']

    def getframe(self):
        ccc = super().getframe()
        luv = ccc * [100.0, 200.0, 200.0] - [0.0, 100.0, 100.0]
        rgb = color.luv2rgb(luv)
        return rgb

# This is supposed to do Lab color but always complains about Zs < 0
class labfields(colorfields):
    colorspace = 'lab'
    colornames = ['l', 'a', 'b']

    def getframe(self):
        ccc = super().getframe()
        lab = ccc * [100.0, 100.0, 100.0] - [0.0, 0.0, 0.0]
        rgb = color.lab2rgb(lab)
        return rgb

# These have an extra reinterpretation and are doubly messed up

class rthfields(colorfields):
    colorspace = 'rth'
    colornames = ['r', 't', 'h']

    def getframe(self):
        ccc = super().getframe()
        xyz = color.rgb2xyz(ccc)
        hsv = color.rgb2hsv(xyz)
        return hsv

class htrfields(colorfields):
    colorspace = 'htr'
    colornames = ['h', 't', 'r']

    def getframe(self):
        ccc = super().getframe()
        hsv = color.rgb2hsv(ccc)
        xyz = color.rgb2xyz(hsv)
        return xyz

# itu = 'hsv' plus one, really hsv twice
class itufields(colorfields):
    colorspace = 'itu'
    colornames = ['ivf', 'tbu', 'ubm']

    def getframe(self):
        hsv = super().getframe()
        rgb = color.hsv2rgb(hsv)
        rgb = color.hsv2rgb(rgb)
        return rgb

# abc = xyz twice
class abcfields(colorfields):
    colorspace = 'abc'
    colornames = ['a', 'b', 'c']

    def getframe(self):
        ccc = super().getframe()
        abc = color.xyz2rgb(ccc)
        rgb = color.xyz2rgb(abc)
        return rgb

from PIL import Image
from PIL import PngImagePlugin

def load_png( infile ):
    data = None
    with Image.open( infile ) as pf:
        data = json.loads( pf.info['Comment'] )
    return data

from hsaudiotag import mp4 # this is not in standard WinPython dist

def load_mp4( infile ):
    data = None
    mf = mp4.File( infile )
    data = json.loads( mf.comment )
    mf.close()
    return data

def load_json( infile ):
    data = None
    with open( infile ) as data_file:
        data = json.load(data_file)
    return data

def load_file( infile ):
    name, ext = os.path.splitext( infile )
    if ext == '.json':
        return load_json( infile )
    elif ext == '.png':
        return load_png( infile )
    elif ext == '.mp4':
        return load_mp4( infile )
    else:
        raise ValueError( 'Unrecognized file extention {}'.format(ext) )

def dozoomer(zoom, xyz):
    """sets zoomer, takes dict zoom and tuple x,y,z,cx,cy"""
    if any(xyz):
        zoomer.set(xyz)
    else:
        ztup = tuple( [ zoom['x'], zoom['y'], zoom['z'], zoom['cx'], zoom['cy'] ] )
        zoomer.set(ztup)

def loadfields( infile, incolor=None, *, xyz ):
    fields = None
    theclass = None
    data = None
    classes = { 'rgb' : rgbfields,
                'hsv' : hsvfields,
                'xyz' : xyzfields,
                'luv' : luvfields,
                'lab' : labfields,
                'rth' : rthfields,
                'itu' : itufields}
                #'abc' : abcfields, 'htr' : htrfields, } # these are boring

    if infile == None:                      # make a random one
        del classes['lab']    # no labfields I don't know valid values
        theclass = random.choice( [v for v in classes.values()] )
        zoomer.set( xyz )       # xyz or default
    else:                           # load data and make appropriate colorfield
        data = load_file( infile )

        if incolor == None:
            cs = data['colorspace']
        else:
            cdat = load_file( incolor )
            cs = cdat['colorspace']
            data['colorspace'] = cs
            data['coloroptions'] = cdat['coloroptions']
            data['wavelist']['guid'] = str(uuid.uuid4())

        theclass = classes[cs]
        dozoomer(data['wavelist']['zoom'], xyz) # xyz or default to data['zoom']
    fields = theclass(data)

    return fields
# end loadfields()

def makefig():
    """Returns a plt.figure() of size xratio:yratio with no axes"""
    fig = plt.figure()
    fig.set_size_inches(zoomer.xratio, zoomer.yratio)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig

def dofilename(guid, outname, outext):
    outfilename = "dofilename() has failed"
    if outname == None:
        outfilename = "{}{}".format(guid, outext)
    else:
        if os.path.exists( outname ):
            if os.path.isdir( outname ):
                os.chdir( outname )
                outfilename = "{}{}".format(guid, outext)
            else:
                raise FileExistsError('{} already exists.'.format( outname ))
        else: # outname does not exist
            noxt, ext = os.path.splitext( outname )
            if len(ext) == 0:   # no extension, guess it is a directory
                os.mkdir( outname )
                os.chdir( outname )
                outfilename = "{}{}".format(guid, outext)
            else:               # it has an extension
                outfilename = outname
    return outfilename

def main_json( infile=None, outname=None, incolor=None, *, xyz ):
    fields = loadfields( infile, incolor, xyz=xyz )
    with open(outname, 'w') as jf:
        jf.write(str(fields))

def main_movie( infile=None, outname=None, incolor=None, *, xyz ):
    # initialize the fields

    fields = loadfields( infile, incolor, xyz=xyz )
    print(fields.guid)
    fig = makefig()
    comtag = str(fields)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=str(fields.guid), comment=comtag, artist='@toddhisattva', composer='https://github.com/ToddHartmann/phaseflower-py')
    writer = FFMpegWriter(fps=mfps, codec='libx264', metadata=metadata)

    outfilename = dofilename(fields.guid, outname, '.mp4')

    tstart = datetime.datetime.now()
    zp = fields.getframe()
    img = plt.imshow(zp)        # just to set img up
    with writer.saving(fig, outfilename, zoomer.dpi):
        fields.cycle(writer, img)

    tdur = datetime.datetime.now() - tstart
    tsec = tdur.total_seconds()
    tfps = mfcount / tsec
    print("Phrames pher sec {}".format(tfps)  )
    print(outfilename)


# http://blog.client9.com/2007/08/28/python-pil-and-png-metadata-take-2.html

def pltsavefig(pngname, commenttag):
    """Saves the figure only once ;-) makes sure it is the right
    size right dpi and has a comment tag."""
    canvas = plt.get_current_fig_manager().canvas
    canvas.figure.set_dpi(zoomer.dpi)
    canvas.draw()
    canstr = canvas.tostring_rgb()
    wh = canvas.get_width_height()

    meta = PngImagePlugin.PngInfo()
    meta.add_text('Comment', commenttag)

    pim = Image.frombytes('RGB', wh, canstr)
    pim.save(pngname, dpi=(zoomer.dpi,zoomer.dpi), pnginfo=meta)
    pim.close()

def thumbnail(*, xyz):
    # initialize the fields
    fields = loadfields(None,xyz=xyz)
    fig = makefig()
    print(fields.guid)
    comtag = str(fields)

    zp = fields.getframe()
    img = plt.imshow(zp)
    rgb = fields.getframe()
    img.set_data( gaussian( rgb, 1.5, multichannel=True ) ) # set multichannel so it doesn't warn about guessing

    pngname = "{}.png".format(fields.guid)
    pltsavefig(pngname, comtag)
    plt.close(fig)

def main_thumb(numb=1, outdir=None, *, xyz):
    if outdir != None:
        if os.path.exists(outdir):
            if os.path.isdir(outdir):
                os.chdir(outdir)
            else:
                raise NotADirectoryError('{} is not a directory.'.format(outdir))
        else: # outdir does not exist, make it and go there
            os.mkdir(outdir)
            os.chdir(outdir)

    for i in range(numb):
        thumbnail(xyz=xyz)

import argparse
import sys            # because argparse can be silly
import textwrap       # because argparse can be quite good

def fillit(s): return textwrap.fill(' '.join(s.split()))

def main():
    depi = '\n'.join( [
        'phaseflower                = make random png w/ default 640:360 zoom=1',
        'phaseflower -n N           = make N random pngs',
        'phaseflower -n N -o outdir = make N random pngs in outdir',
        'phaseflower -y 1080        = make 1 random png w/ height=1080',
        'phaseflower -s S           = make random movie S seconds long',
        'phaseflower -r R           = make random movie w/ framerate R',
        'phaseflower -s S -y 1080   = make random movie S seconds long and 1080 pixels high',
        'phaseflower -i name.ext    = make movie from input file (.json, .mp4, or .png)',
        'phaseflower -i waves.ext -ic color.ext',
        '                    = make movie using waves from -i and colors from -ic',
        'phaseflower -o outdir      = make random movie in outdir',
        'phaseflower -o out.mp4     = make random movie in out.mp4',
        'phaseflower -i name.ext -oj outname.json',
        '                    = output the JSON to a file and exit',
        'phaseflower -oj name.ext   = make a JSON file with random waves and colors',
        '\nYou cannot make thumbnails and movies at the same time.  Neither can',
        'you make thumbnails and JSON at the same time (you get thumbnails),',
        'nor JSON and movies at the same time (you get JSON).  Other than',
        'those restrictions, you can combine almost all the options, the',
        'exceptions are few and make sense.'
        '\n\nThe default aspect ratio is 16:9 if either -x or -y is missing.'
        ] )

    parser = argparse.ArgumentParser( description = 'Make nice video loops.', epilog = depi,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', required=False, help='Input file')
    parser.add_argument('-o', required=False, help='Output file or directory')
    parser.add_argument('-ic', required=False, help='File to load color info from')
    parser.add_argument('-oj', required=False, help='File to output JSON to and exit')
    parser.add_argument('-n', required=False, type=int, default=1, help='Number of thumbnails to make')
    parser.add_argument('-s', required=False, type=float, default=4,  help='Length of movie in seconds (default 4), negative to go backwards')
    parser.add_argument('-r', required=False, type=float, default=30, help='Rate in frames per second (default 30)')
    parser.add_argument('-ef', required=False, type=int, default=0, help='Number of extra frames to add or subtract (total = S * R + EF) ')
    parser.add_argument('-x', required=False, type=int, help='Width in pixels (default 360)')
    parser.add_argument('-y', required=False, type=int, help='Height in pixels (default 640)')
    parser.add_argument('-z', required=False, type=float, help='Zoom (2=plane half size, .5=2x)')
    parser.add_argument('-cx', required=False, type=float, help='Center x-coord in radians')
    parser.add_argument('-cy', required=False, type=float, help='Center y-coord in radians')

    args = parser.parse_args()
    xyz  = ( args.x, args.y, args.z, args.cx, args.cy )
    lxyz = len(xyz) - xyz.count(None)
    lsa  = len(sys.argv) - lxyz * 2   # how long is the arg list without any -x X -y Y -z Z
    if lsa == 1:      # no other args sent from command line, random thumb
        main_thumb( numb=1, xyz=xyz )    # or do we want to make a random main_movie()?
    else:
        if '-n' in sys.argv:                     # -n forces thumbnail mode
            main_thumb( numb=args.n, outdir=args.o, xyz=xyz )
        else:
            setlength( args.s, args.r, args.ef )
            if '-oj' in sys.argv:   # JSON output
                main_json( infile=args.i, outname=args.oj, incolor=args.ic, xyz=xyz )
            else:            # movie mode
                main_movie( infile=args.i, outname=args.o, incolor=args.ic, xyz=xyz )

"""
import cProfile, pstats, io

def statme(pr):
    s = io.StringIO()
    sortby = 'tottime' #'cumulative'
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats(20)#'phaseflower.py:')
    print(s.getvalue())
"""

if __name__ == '__main__':
#    pr = cProfile.Profile()
#    pr.enable()
    main()
#    pr.disable()
    #statme(pr)

    print("PhaseFlowEr Finished.")
    pass
