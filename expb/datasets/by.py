import typing

ByOption = typing.NewType("ByOption", str)


class By:
    CATEGORY = ByOption("category")
    IMGSHAPE = ByOption("img_shape")
    TAG = ByOption("tag")
