"""Smoothed density of scanpath points, in original image coordinates."""
import math
from PySide6.QtGui import QColor, QImage

_COLORS = [QColor.fromHsvF((1-i/255)*.64, .95, 1., min(1., i/255*2)).rgba() for i in range(256)]


class HeatmapDensity:
    def __init__(self, width, height):
        self.scale = min(1., 200/max(width, height))
        self.w, self.h = max(1, round(width*self.scale)), max(1, round(height*self.scale))
        self.sigma = max(2., min(width, height)*.045*self.scale)
        self.values = [0.]*(self.w*self.h)
        self.count = 0

    def add(self, points):
        radius = math.ceil(3*self.sigma)
        denominator = 2*self.sigma*self.sigma
        for point in points:
            cx, cy = point['x']*self.scale, point['y']*self.scale
            xs = [(x, math.exp(-(x-cx)**2/denominator))
                  for x in range(max(0,int(cx)-radius), min(self.w,int(cx)+radius+1))]
            for y in range(max(0,int(cy)-radius), min(self.h,int(cy)+radius+1)):
                wy = math.exp(-(y-cy)**2/denominator)
                offset = y*self.w
                for x, wx in xs:
                    self.values[offset+x] += wx*wy
            self.count += 1

    def image(self):
        if not self.count:
            return QImage()
        image = QImage(self.w, self.h, QImage.Format.Format_ARGB32)
        image.fill(0)
        peak = max(self.values)
        if peak:
            for i, value in enumerate(self.values):
                level = value/peak
                if level > .015:
                    image.setPixel(i % self.w, i // self.w, _COLORS[min(255, round(level*255))])
        return image


def route_heatmap(points, width, height):
    density = HeatmapDensity(width, height)
    density.add(points)
    return density.image()
