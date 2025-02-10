import typing

LabelingPlatformOption = typing.NewType("LabelingPlatformOption", str)


class LabelingPlatform(object):
    ROBOFLOW = LabelingPlatformOption("roboflow")
