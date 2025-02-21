import typing

ByOption = typing.NewType("ByOption", str)


class By:
    CATEGORY = ByOption("category")
    IMG_SHAPE = ByOption("img_shape")
    TAG = ByOption("tag")
