# Phaseflower

Make nice MP4 video loops.

Works well in [Anaconda](https://www.anaconda.com/)
but any well-stocked [SciPy](https://scipy.org/) install should do.

Author: [Todd Hartmann](https://github.com/ToddHartmann)\
License:  Public Domain, Use At Your Own Risk

## The Help
```
usage: phaseflower.py [-h] [-i I] [-o O] [-ic IC] [-oj OJ] [-n N] [-s S]
                      [-r R] [-ef EF] [-x X] [-y Y] [-z Z] [-cx CX] [-cy CY]

Make nice video loops.

optional arguments:
  -h, --help  show this help message and exit
  -i I        Input file
  -o O        Output file or directory
  -ic IC      File to load color info from
  -oj OJ      File to output JSON to and exit
  -n N        Number of thumbnails to make
  -s S        Length of movie in seconds (default 4), negative to go backwards
  -r R        Rate in frames per second (default 30)
  -ef EF      Number of extra frames to add or subtract (total = S * R + EF)
  -x X        Width in pixels (default 360)
  -y Y        Height in pixels (default 640)
  -z Z        Zoom (2=plane half size, .5=2x)
  -cx CX      Center x-coord in math
  -cy CY      Center y-coord in math

phaseflower                = make random png w/ default 640:360 zoom=1
phaseflower -n N           = make N random pngs
phaseflower -n N -o outdir = make N random pngs in outdir
phaseflower -y 1080        = make 1 random png w/ height=1080
phaseflower -s S           = make random movie S seconds long
phaseflower -r R           = make random movie w/ framerate R
phaseflower -s S -y 1080   = make random movie S seconds long and 1080 pixels high
phaseflower -i name.ext    = make movie from input file (.json, .mp4, or .png)
phaseflower -i waves.ext -ic color.ext
                    = make movie using waves from -i and colors from -ic
phaseflower -o outdir      = make random movie in outdir
phaseflower -o out.mp4     = make random movie in out.mp4
phaseflower -i name.ext -oj outname.json
                    = output the JSON to a file and exit
phaseflower -oj name.ext   = make a JSON file with random waves and colors

You cannot make thumbnails and movies at the same time.  Neither can
you make thumbnails and JSON at the same time (you get thumbnails),
nor JSON and movies at the same time (you get JSON).  Other than
those restrictions, you can combine almost all the options, the
exceptions are few and make sense.

The default aspect ratio is 16:9 if either -x or -y is missing.
```
## Examples


[![Simple](https://img.youtube.com/vi/7Hb88SFlkUY/0.jpg)](https://www.youtube.com/watch?v=7Hb88SFlkUY)
[simple.json](examples/simple.json)
```
{
    "colorspace": "rgb",
    "coloroptions": [["red", "green", "blue"], [1, 1, 1], [false, false, false]],
    "wavelist":
    {
        "zoom": {"cx": 0.0, "cy": 0.0, "x": 512, "y": 512, "z": 1.0},
        "guid": "b911419e-54e5-4d8f-a758-d17df9d993fb",
        "waves":
        [
            {"amp": 0.999999, "cx": 0.0, "cy": 0.0, "exp": 1, "freq": 1, "phase": 0.0}
        ]
    }
}
```

[![Simple](https://img.youtube.com/vi/xdwipXjREdA/0.jpg)](https://www.youtube.com/watch?v=xdwipXjREdA)
[smrat.json](examples/smrat.json)

```
{
    "colorspace": "rgb",
    "coloroptions": [["red", "green", "blue"], [8, 2, 8], [false, false, true]],
    "wavelist":
    {
        "zoom": {"cx": 0.0, "cy": 0.0, "x": 640, "y": 360, "z": 1.0},
        "guid": "eb5f9cf4-9541-4e1f-b04e-44ceadc893fa",
        "waves":
        [
            {"amp": 0.356725, "cx": 2.008187, "cy": -3.356086, "exp": 1, "freq": 3, "phase": 0.996851},
            {"amp": 0.530763, "cx": 2.924109, "cy": 5.259152, "exp": 3, "freq": 2, "phase": 3.51363},
            {"amp": 0.479494, "cx": -4.08501, "cy": -4.258618, "exp": 1, "freq": 2, "phase": 0.788228},
            {"amp": 0.461836, "cx": 4.223394, "cy": -4.435429, "exp": 1, "freq": -2, "phase": 0.650822}
        ]
    }
}
```

[![Greenrose](https://img.youtube.com/vi/QxRH4npChq4/0.jpg)](https://www.youtube.com/watch?v=QxRH4npChq4)
[grose.json](examples/grose.json)
```
{
    "colorspace": "luv",
    "coloroptions": [["l", "u", "v"], [8, 1, 2], [true, false, false]],
    "wavelist":
    {
        "zoom": {"cx": 0.0, "cy": 0.0, "x": 640, "y": 360, "z": 1.0},
        "guid": "78b5e1d7-7a1c-4b2e-bf1b-80f788958707",
        "waves":
        [
            {"amp": 0.281749, "cx": -2.982593, "cy": -2.568968, "exp": 1, "freq": 4, "phase": 5.193345},
            {"amp": 0.784343, "cx": -2.451323, "cy": 1.162091, "exp": 1, "freq": -4, "phase": 4.009389},
            {"amp": 0.917097, "cx": -7.858992, "cy": -5.06226, "exp": 3, "freq": 1, "phase": 0.4588},
            {"amp": 0.517462, "cx": -7.698834, "cy": 0.771917, "exp": 7, "freq": 4, "phase": 0.43838},
            {"amp": 0.506322, "cx": 6.590463, "cy": 2.998623, "exp": 1, "freq": -4, "phase": 3.157555}
        ]
    }
}
```
