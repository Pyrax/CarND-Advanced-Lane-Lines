import numpy as np


class ImageProcessingPipeline:
    def __init__(self):
        self.steps = []
        return

    def register_step(self, step_function, *args, **kwargs):
        self.steps.append({"handler": step_function, "args": args, "kwargs": kwargs})
        return self

    def register_steps(self, steps):
        for step in steps:
            self.register_step(step)
        return self

    """ Run pipeline steps on a given input image. """
    def run(self, image):
        dest_image = np.copy(image)
        for step in self.steps:
            handler, args, kwargs = step["handler"], step["args"], step["kwargs"]
            dest_image = handler(dest_image, *args, **kwargs)
        return dest_image
