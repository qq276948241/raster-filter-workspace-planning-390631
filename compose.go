package gift

import (
	"image"
	"image/draw"
)

// composeOver places tmp on top of dst using the source-over composition.
// tb is the bounds of tmp in the destination coordinate system. Pixels outside
// the destination bounds are clipped. The pixel math is identical to the
// GIFT.DrawAt implementation.
func composeOver(dst draw.Image, tmp image.Image, tb image.Rectangle, options *Options) {
	if options == nil {
		options = &defaultOptions
	}
	pixGetterDst := newPixelGetter(dst)
	pixGetterTmp := newPixelGetter(tmp)
	pixSetterDst := newPixelSetter(dst)
	ib := tb.Intersect(dst.Bounds())
	parallelize(options.Parallelization, ib.Min.Y, ib.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := ib.Min.X; x < ib.Max.X; x++ {
				px0 := pixGetterDst.getPixel(x, y)
				px1 := pixGetterTmp.getPixel(x, y)
				c1 := px1.a
				c0 := (1 - c1) * px0.a
				cs := c0 + c1
				c0 /= cs
				c1 /= cs
				r := px0.r*c0 + px1.r*c1
				g := px0.g*c0 + px1.g*c1
				b := px0.b*c0 + px1.b*c1
				a := px0.a + px1.a*(1-px0.a)
				pixSetterDst.setPixel(x, y, pixel{r, g, b, a})
			}
		}
	})
}

// composeCopy replaces the dst pixels with tmp pixels. tb is the bounds of tmp
// in the destination coordinate system. Pixels outside the destination bounds
// are clipped. The pixel math is identical to the GIFT.DrawAt implementation.
func composeCopy(dst draw.Image, tmp image.Image, tb image.Rectangle, options *Options) {
	if options == nil {
		options = &defaultOptions
	}
	pixGetter := newPixelGetter(tmp)
	pixSetter := newPixelSetter(dst)
	ib := tb.Intersect(dst.Bounds())
	parallelize(options.Parallelization, ib.Min.Y, ib.Max.Y, func(start, stop int) {
		for y := start; y < stop; y++ {
			for x := ib.Min.X; x < ib.Max.X; x++ {
				pixSetter.setPixel(x, y, pixGetter.getPixel(x, y))
			}
		}
	})
}
