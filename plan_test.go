package gift

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
	"math"
	"runtime"
	"sync"
	"testing"
)

// newProceduralImage builds a small deterministic image with a non-zero origin
// and partly transparent pixels. The pixel value is derived from the
// coordinates so every position is distinguishable.
func newProceduralImage(origin image.Point, w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rectangle{
		Min: origin,
		Max: origin.Add(image.Pt(w, h)),
	})
	for y := img.Rect.Min.Y; y < img.Rect.Max.Y; y++ {
		for x := img.Rect.Min.X; x < img.Rect.Max.X; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = uint8((x*7 + y*3) & 0xff)
			img.Pix[i+1] = uint8((x*5 + y*11) & 0xff)
			img.Pix[i+2] = uint8((x*13 + y*17) & 0xff)
			img.Pix[i+3] = uint8(64 + ((x+y)*9)&127)
		}
	}
	return img
}

// assertPixelsEqual compares two images pixel by pixel over their bounds.
func assertPixelsEqual(t *testing.T, got, want image.Image, prefix string) {
	t.Helper()
	gb, wb := got.Bounds(), want.Bounds()
	if !gb.Size().Eq(wb.Size()) {
		t.Fatalf("%s: size mismatch got=%v want=%v", prefix, gb.Size(), wb.Size())
	}
	for y := 0; y < gb.Dy(); y++ {
		for x := 0; x < gb.Dx(); x++ {
			gp := got.At(gb.Min.X+x, gb.Min.Y+y)
			wp := want.At(wb.Min.X+x, wb.Min.Y+y)
			if gp != wp {
				t.Fatalf("%s: pixel (%d,%d) mismatch got=%v want=%v", prefix, x, y, gp, wp)
			}
		}
	}
}

// testChains covers resize, crop, rotation (with and without interpolation),
// geometry transforms, color work and transparent composition in one chain.
func testChains() [][]Filter {
	return [][]Filter{
		{Resize(12, 0, BoxResampling), Grayscale(), Rotate90()},
		{Crop(image.Rect(1, 1, 6, 5)), Resize(10, 8, LinearResampling), Rotate(30, color.Transparent, NearestNeighborInterpolation)},
		{Rotate(45, color.NRGBA{R: 10, G: 20, B: 30, A: 200}, LinearInterpolation), Resize(14, 14, CubicResampling), FlipHorizontal()},
		{Resize(20, 16, LanczosResampling), CropToSize(12, 10, CenterAnchor), Transpose()},
		{Rotate180(), Resize(9, 9, NearestNeighborResampling), Invert()},
		{GaussianBlur(1.2), Resize(8, 12, BoxResampling), Contrast(15)},
	}
}

func TestPlanPixelParity(t *testing.T) {
	origins := []image.Point{
		image.Pt(0, 0),
		image.Pt(-7, -3),
		image.Pt(5, 9),
	}
	for ci, chain := range testChains() {
		for _, origin := range origins {
			src := newProceduralImage(origin, 13, 9)

			g := New(chain...)
			oldDst := image.NewNRGBA64(g.Bounds(src.Bounds()))
			g.Draw(oldDst, src)

			plan, err := g.Compile(src.Bounds())
			if err != nil {
				t.Fatalf("chain %d origin %v: compile: %v", ci, origin, err)
			}
			if !plan.OutputBounds().Eq(g.Bounds(src.Bounds())) {
				t.Fatalf("chain %d: output bounds %v != gift bounds %v",
					ci, plan.OutputBounds(), g.Bounds(src.Bounds()))
			}

			ws, err := NewWorkspace(math.MaxInt64)
			if err != nil {
				t.Fatal(err)
			}
			newDst := image.NewNRGBA64(plan.OutputBounds())
			if err := plan.Draw(ws, newDst, src); err != nil {
				t.Fatalf("chain %d origin %v: draw: %v", ci, origin, err)
			}
			assertPixelsEqual(t, newDst, oldDst, "chain parity")

			// Also check an NRGBA destination and a non-zero dst origin.
			shiftedDst := image.NewNRGBA(plan.OutputBounds().Add(image.Pt(2, 4)))
			oldShifted := image.NewNRGBA(g.Bounds(src.Bounds()).Add(image.Pt(2, 4)))
			g.Draw(oldShifted, src)
			if err := plan.Draw(ws, shiftedDst, src); err != nil {
				t.Fatalf("chain %d: shifted draw: %v", ci, err)
			}
			assertPixelsEqual(t, shiftedDst, oldShifted, "shifted dst parity")
			ws.Release()
		}
	}
}

func TestPlanEmptyChain(t *testing.T) {
	src := newProceduralImage(image.Pt(-4, 2), 7, 5)
	g := New()
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	if plan.RequiredPixels() != 0 {
		t.Fatalf("empty chain needs %d pixels, want 0", plan.RequiredPixels())
	}
	ws, err := NewWorkspace(0)
	if err != nil {
		t.Fatal(err)
	}
	// The empty chain copies src into dst using dst's Min as the origin, so the
	// destination carries the same size and origin as the source.
	dst := image.NewNRGBA64(image.Rectangle{Min: src.Bounds().Min, Max: src.Bounds().Min.Add(src.Bounds().Size())})
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatal(err)
	}
	oldDst := image.NewNRGBA64(src.Bounds())
	g.Draw(oldDst, src)
	assertPixelsEqual(t, dst, oldDst, "empty chain")
}

func TestPlanEmptyImage(t *testing.T) {
	g := New(Resize(10, 10, BoxResampling), Grayscale())
	emptyBounds := image.Rect(-3, -3, -3, -3)
	plan, err := g.Compile(emptyBounds)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.OutputBounds().Empty() {
		t.Fatalf("empty source produced non-empty output %v", plan.OutputBounds())
	}
	ws, err := NewWorkspace(0)
	if err != nil {
		t.Fatal(err)
	}
	src := image.NewNRGBA(emptyBounds)
	dst := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	want := dst.Pix
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if dst.Pix[i] != want[i] {
			t.Fatalf("empty source modified destination pixel %d", i)
		}
	}
}

func TestPlanDrawAtParityAndClipping(t *testing.T) {
	src := newProceduralImage(image.Pt(-2, -1), 11, 8)
	chain := []Filter{Resize(9, 7, LinearResampling), Rotate90()}
	g := New(chain...)
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}

	type caseDef struct {
		pt image.Point
		op Operator
	}
	cases := []caseDef{
		{image.Pt(0, 0), CopyOperator},
		{image.Pt(3, -2), CopyOperator},
		{image.Pt(-4, 2), OverOperator},
		{image.Pt(6, 5), OverOperator},
		{image.Pt(-20, -20), CopyOperator}, // fully outside
	}
	required, err := plan.RequiredDrawAtPixels()
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(required)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		oldDst := image.NewNRGBA64(image.Rect(-5, -5, 15, 15))
		fillMarker(oldDst)
		g.DrawAt(oldDst, src, c.pt, c.op)

		newDst := image.NewNRGBA64(image.Rect(-5, -5, 15, 15))
		fillMarker(newDst)
		if err := plan.DrawAt(ws, newDst, src, c.pt, c.op); err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		assertPixelsEqual(t, newDst, oldDst, "drawat parity")
	}
}

// fillMarker fills an image with a sentinel color so untouched pixels are
// observable.
func fillMarker(img *image.NRGBA64) {
	for i := 0; i < len(img.Pix); i += 8 {
		img.Pix[i+0] = 0x11
		img.Pix[i+1] = 0x22
		img.Pix[i+2] = 0x33
		img.Pix[i+3] = 0x44
		img.Pix[i+4] = 0x55
		img.Pix[i+5] = 0x66
		img.Pix[i+6] = 0x77
		img.Pix[i+7] = 0x88
	}
}

func TestPlanDrawAtBudgetAndOffset(t *testing.T) {
	src := newProceduralImage(image.Pt(-3, -2), 9, 7)
	chain := []Filter{Resize(8, 6, BoxResampling), FlipVertical(), Rotate90()}
	g := New(chain...)
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	need, err := plan.RequiredDrawAtPixels()
	if err != nil {
		t.Fatal(err)
	}

	// One pixel short of the DrawAt requirement: error, destination untouched.
	wsShort, err := NewWorkspace(need - 1)
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(image.Rect(-4, -4, 12, 12))
	fillMarker(dst)
	want := append([]uint8(nil), dst.Pix...)
	if err := plan.DrawAt(wsShort, dst, src, image.Pt(-2, 1), OverOperator); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded, got %v", err)
	}
	if string(dst.Pix) != string(want) {
		t.Fatal("destination modified after DrawAt budget failure")
	}

	// Exact budget, negative-origin placement clipped against dst.
	wsExact, err := NewWorkspace(need)
	if err != nil {
		t.Fatal(err)
	}
	oldDst := image.NewNRGBA64(image.Rect(-4, -4, 12, 12))
	fillMarker(oldDst)
	fillMarker(dst)
	g.DrawAt(oldDst, src, image.Pt(-2, 1), OverOperator)
	if err := plan.DrawAt(wsExact, dst, src, image.Pt(-2, 1), OverOperator); err != nil {
		t.Fatal(err)
	}
	assertPixelsEqual(t, dst, oldDst, "drawat offset over")

	// Fully clipped placement is a no-op even with a zero-budget workspace.
	wsZero, err := NewWorkspace(0)
	if err != nil {
		t.Fatal(err)
	}
	dst2 := image.NewNRGBA64(image.Rect(0, 0, 3, 3))
	fillMarker(dst2)
	want2 := append([]uint8(nil), dst2.Pix...)
	if err := plan.DrawAt(wsZero, dst2, src, image.Pt(100, 100), CopyOperator); err != nil {
		t.Fatal(err)
	}
	if string(dst2.Pix) != string(want2) {
		t.Fatal("fully clipped DrawAt modified the destination")
	}
}

func TestPlanValidationLeavesDstUntouched(t *testing.T) {
	src := newProceduralImage(image.Pt(0, 0), 10, 10)
	g := New(Resize(12, 12, BoxResampling), Grayscale(), Rotate90())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(plan.RequiredPixels())
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(plan.OutputBounds())
	fillMarker(dst)
	want := append([]uint8(nil), dst.Pix...)

	// Wrong source size.
	bigSrc := newProceduralImage(image.Pt(0, 0), 11, 10)
	if err := plan.Draw(ws, dst, bigSrc); !errors.Is(err, ErrSourceSizeMismatch) {
		t.Fatalf("want ErrSourceSizeMismatch, got %v", err)
	}
	if string(dst.Pix) != string(want) {
		t.Fatal("destination modified after source size failure")
	}

	// Destination too small.
	smallDst := image.NewNRGBA64(image.Rect(0, 0, plan.OutputBounds().Dx()-1, plan.OutputBounds().Dy()))
	if err := plan.Draw(ws, smallDst, src); !errors.Is(err, ErrDstTooSmall) {
		t.Fatalf("want ErrDstTooSmall, got %v", err)
	}

	// Budget one pixel short.
	poor, err := NewWorkspace(plan.RequiredPixels() - 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.Draw(poor, dst, src); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded, got %v", err)
	}
	if string(dst.Pix) != string(want) {
		t.Fatal("destination modified after budget failure")
	}

	// Nil arguments.
	if err := plan.Draw(nil, dst, src); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("want ErrInvalidArgument, got %v", err)
	}
	if _, err := NewWorkspace(-1); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("negative budget: want ErrBudgetExceeded, got %v", err)
	}

	// After all the validation failures the same workspace still executes
	// successfully and no occupancy marker was retained.
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatalf("workspace not reusable after validation failures: %v", err)
	}
	oldDst := image.NewNRGBA64(g.Bounds(src.Bounds()))
	g.Draw(oldDst, src)
	assertPixelsEqual(t, dst, oldDst, "post-failure reuse")
}

func TestPlanBudgetExactAndOnePixelShort(t *testing.T) {
	src := newProceduralImage(image.Pt(0, 0), 8, 6)
	// Three stages -> two temps; plan.RequiredPixels() is the exact peak.
	g := New(
		Resize(10, 9, BoxResampling),
		Resize(7, 11, LinearResampling),
		Grayscale(),
	)
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	need := plan.RequiredPixels()
	if need <= 0 {
		t.Fatalf("unexpected required pixels: %d", need)
	}

	wsExact, err := NewWorkspace(need)
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(plan.OutputBounds())
	if err := plan.Draw(wsExact, dst, src); err != nil {
		t.Fatalf("exact budget draw failed: %v", err)
	}
	oldDst := image.NewNRGBA64(g.Bounds(src.Bounds()))
	g.Draw(oldDst, src)
	assertPixelsEqual(t, dst, oldDst, "exact budget")

	wsShort, err := NewWorkspace(need - 1)
	if err != nil {
		t.Fatal(err)
	}
	dst2 := image.NewNRGBA64(plan.OutputBounds())
	if err := plan.Draw(wsShort, dst2, src); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded with budget %d (need %d), got %v", need-1, need, err)
	}
}

func TestPlanReuseAfterFailure(t *testing.T) {
	src := newProceduralImage(image.Pt(0, 0), 8, 8)
	g := New(Resize(12, 10, BoxResampling), Rotate90(), Grayscale())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(plan.RequiredPixels() / 2)
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(plan.OutputBounds())

	// Failing executions must leave no occupancy markers behind.
	for i := 0; i < 5; i++ {
		if err := plan.Draw(ws, dst, src); !errors.Is(err, ErrBudgetExceeded) {
			t.Fatalf("iteration %d: want budget error, got %v", i, err)
		}
	}
	ws.mu.Lock()
	if ws.busy {
		t.Fatal("workspace left busy after failed execution")
	}
	if ws.live != 0 {
		t.Fatalf("workspace live pixels = %d after failure, want 0", ws.live)
	}
	ws.mu.Unlock()

	// The plan remains usable for the next image through another workspace, and
	// the failed workspace did not keep any occupancy marker.
	good, err := NewWorkspace(plan.RequiredPixels())
	if err != nil {
		t.Fatal(err)
	}
	okDst := image.NewNRGBA64(plan.OutputBounds())
	if err := plan.Draw(good, okDst, src); err != nil {
		t.Fatalf("plan not reusable after prior failures: %v", err)
	}
	oldDst := image.NewNRGBA64(g.Bounds(src.Bounds()))
	g.Draw(oldDst, src)
	assertPixelsEqual(t, okDst, oldDst, "reuse after failure")
}

func TestWorkspaceRuntimePeakMatchesPlan(t *testing.T) {
	// Instrument the workspace through its mutex to observe the live pixel
	// counter while a chain with overlapping temps runs.
	src := newProceduralImage(image.Pt(0, 0), 8, 6)
	g := New(
		Resize(10, 9, BoxResampling),
		Resize(9, 10, LinearResampling),
		Grayscale(),
	)
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(plan.RequiredPixels())
	if err != nil {
		t.Fatal(err)
	}

	var observedPeak int64
	var peakMu sync.Mutex
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
				ws.mu.Lock()
				cur := ws.live
				ws.mu.Unlock()
				peakMu.Lock()
				if cur > observedPeak {
					observedPeak = cur
				}
				peakMu.Unlock()
			}
		}
	}()

	dst := image.NewNRGBA64(plan.OutputBounds())
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatal(err)
	}
	close(stop)
	<-done
	if observedPeak > plan.RequiredPixels() {
		t.Fatalf("runtime peak %d exceeds compiled requirement %d", observedPeak, plan.RequiredPixels())
	}
	if observedPeak != plan.RequiredPixels() {
		t.Fatalf("runtime peak %d, want %d (budget tightness not exercised)", observedPeak, plan.RequiredPixels())
	}
}

func TestPlanStageBoundsAndPeak(t *testing.T) {
	src := newProceduralImage(image.Pt(-5, 3), 8, 6)
	g := New(Resize(10, 9, BoxResampling), Rotate90(), Grayscale())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	if plan.Len() != 3 {
		t.Fatalf("Len = %d, want 3", plan.Len())
	}
	// Recompute the expected stage cascade exactly like GIFT does.
	b := src.Bounds()
	for i, f := range g.Filters {
		in, out := plan.StageBounds(i)
		if !in.Eq(b) {
			t.Fatalf("stage %d: in bounds %v, want %v", i, in, b)
		}
		b = f.Bounds(b)
		if !out.Eq(b) {
			t.Fatalf("stage %d: out bounds %v, want %v", i, out, b)
		}
	}
	// Rotate90 swaps dimensions: temps are 10x9 then 9x10; the last output
	// (9x10) is the destination and must not be counted. When the second temp
	// is acquired the first one is still the pipeline input, so the peak of
	// simultaneously alive temps is 10*9 + 9*10.
	if want := int64(10*9 + 9*10); plan.RequiredPixels() != want {
		t.Fatalf("RequiredPixels = %d, want %d", plan.RequiredPixels(), want)
	}
	drawAtNeed, err := plan.RequiredDrawAtPixels()
	if err != nil {
		t.Fatal(err)
	}
	if drawAtNeed != plan.RequiredPixels()+int64(9*10) {
		t.Fatalf("DrawAt pixels = %d", drawAtNeed)
	}
}

func TestPlanSnapshotIsolation(t *testing.T) {
	src := newProceduralImage(image.Pt(0, 0), 8, 8)
	g := New(Resize(6, 6, BoxResampling), Grayscale())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	parSnap := plan.Options().Parallelization

	// Mutate the original filter list and options after compilation.
	g.Add(Invert())
	g.Empty()
	g.SetParallelization(!parSnap)

	if plan.Options().Parallelization != parSnap {
		t.Fatal("plan options snapshot changed after SetParallelization")
	}
	if len(plan.stages) != 2 {
		t.Fatalf("plan stages = %d, want 2 after source list mutation", len(plan.stages))
	}
	ws, err := NewWorkspace(math.MaxInt64)
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(plan.OutputBounds())
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatal(err)
	}
	// Rebuild the snapshotted chain independently and compare pixels.
	g2 := New(Resize(6, 6, BoxResampling), Grayscale())
	g2.SetParallelization(parSnap)
	want := image.NewNRGBA64(g2.Bounds(src.Bounds()))
	g2.Draw(want, src)
	assertPixelsEqual(t, dst, want, "snapshot isolation")
}

func TestWorkspaceBufferReuseAndRelease(t *testing.T) {
	src := newProceduralImage(image.Pt(0, 0), 10, 10)
	g := New(Resize(14, 12, BoxResampling), Resize(9, 15, LinearResampling), Grayscale())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(math.MaxInt64)
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(plan.OutputBounds())

	runtime.GC()
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatal(err)
	}
	firstAllocs := ws.allocations
	if firstAllocs == 0 {
		t.Fatal("first execution allocated no workspace buffers")
	}

	// Hot reuse: no new backing arrays.
	for i := 0; i < 4; i++ {
		if err := plan.Draw(ws, dst, src); err != nil {
			t.Fatal(err)
		}
	}
	if ws.allocations != firstAllocs {
		t.Fatalf("hot execution allocated %d new buffers", ws.allocations-firstAllocs)
	}
	if ws.RetainedPixels() == 0 {
		t.Fatal("workspace retained no buffers after execution")
	}

	// Explicit release drops everything.
	ws.Release()
	if ws.RetainedPixels() != 0 {
		t.Fatal("Release did not free retained buffers")
	}
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatalf("workspace unusable after Release: %v", err)
	}
	if ws.allocations == firstAllocs {
		t.Fatal("no reallocation happened after Release")
	}
}

func TestWorkspaceDoesNotRetainOversizedBuffer(t *testing.T) {
	// A one-off large image must not pin a large buffer once smaller images
	// become the regular workload.
	bigSrc := newProceduralImage(image.Pt(0, 0), 30, 30)
	smallSrc := newProceduralImage(image.Pt(0, 0), 8, 8)

	g := New(Resize(0, 20, BoxResampling), Grayscale())
	bigPlan, err := g.Compile(bigSrc.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	smallG := New(Resize(0, 10, BoxResampling), Grayscale())
	smallPlan, err := smallG.Compile(smallSrc.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(math.MaxInt64)
	if err != nil {
		t.Fatal(err)
	}
	if err := bigPlan.Draw(ws, image.NewNRGBA64(bigPlan.OutputBounds()), bigSrc); err != nil {
		t.Fatal(err)
	}
	retainedAfterBig := ws.RetainedPixels()
	if retainedAfterBig == 0 {
		t.Fatal("big run retained no buffers")
	}

	for i := 0; i < 3; i++ {
		if err := smallPlan.Draw(ws, image.NewNRGBA64(smallPlan.OutputBounds()), smallSrc); err != nil {
			t.Fatal(err)
		}
	}
	retainedAfterSmall := ws.RetainedPixels()
	if retainedAfterSmall >= retainedAfterBig {
		t.Fatalf("oversized buffer retained: %d pixels kept after switching to %d-pixel workload",
			retainedAfterSmall, retainedAfterBig)
	}
}

func TestWorkspaceConcurrentSamePlan(t *testing.T) {
	src := newProceduralImage(image.Pt(-3, 2), 12, 9)
	g := New(Resize(10, 10, LanczosResampling), Rotate90(), Grayscale())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}

	const workers = 8
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ws, werr := NewWorkspace(math.MaxInt64)
			if werr != nil {
				errCh <- werr
				return
			}
			for j := 0; j < 5; j++ {
				dst := image.NewNRGBA64(plan.OutputBounds())
				if err := plan.Draw(ws, dst, src); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

func TestWorkspaceBusyOnConcurrentUse(t *testing.T) {
	src := newProceduralImage(image.Pt(0, 0), 16, 16)
	g := New(GaussianBlur(2.0), Resize(10, 10, BoxResampling))
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(math.MaxInt64)
	if err != nil {
		t.Fatal(err)
	}

	// Custom filter blocking inside Draw gives the second goroutine a
	// deterministic window to observe the busy flag.
	started := make(chan struct{})
	release := make(chan struct{})
	blocking := &blockingFilter{started: started, release: release}
	blockPlan, err := New(blocking, Grayscale()).Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_ = blockPlan.Draw(ws, image.NewNRGBA64(blockPlan.OutputBounds()), src)
	}()
	go func() {
		defer wg.Done()
		<-started
		err := plan.Draw(ws, image.NewNRGBA64(plan.OutputBounds()), src)
		if !errors.Is(err, ErrWorkspaceBusy) {
			t.Errorf("want ErrWorkspaceBusy, got %v", err)
		}
		close(release)
	}()
	wg.Wait()

	// After the blocked execution finishes the workspace is usable again.
	if err := plan.Draw(ws, image.NewNRGBA64(plan.OutputBounds()), src); err != nil {
		t.Fatalf("workspace not recoverable: %v", err)
	}
}

type blockingFilter struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (f *blockingFilter) Bounds(srcBounds image.Rectangle) image.Rectangle {
	return image.Rect(0, 0, srcBounds.Dx(), srcBounds.Dy())
}

func (f *blockingFilter) Draw(dst draw.Image, src image.Image, options *Options) {
	f.once.Do(func() { close(f.started) })
	<-f.release
	copyimage(dst, src, options)
}

func TestPlanOverflow(t *testing.T) {
	// A filter reporting impossibly large bounds must fail at compile time.
	huge := &hugeBoundsFilter{}
	if _, err := New(huge).Compile(image.Rect(0, 0, 4, 4)); !errors.Is(err, ErrOverflow) {
		t.Fatalf("want ErrOverflow, got %v", err)
	}

	// Non-canonical source bounds (image.Rect canonicalizes, so build the
	// inverted rectangle literally).
	invalidRect := image.Rectangle{Min: image.Pt(0, 0), Max: image.Pt(-1, 4)}
	if _, err := New(Grayscale()).Compile(invalidRect); !errors.Is(err, ErrInvalidBounds) {
		t.Fatalf("want ErrInvalidBounds, got %v", err)
	}
}

type hugeBoundsFilter struct{}

func (f *hugeBoundsFilter) Bounds(srcBounds image.Rectangle) image.Rectangle {
	// Each side fits in int but the width*height product overflows int, so
	// Compile's checked multiplication is what rejects it.
	maxInt := int(^uint(0) >> 1)
	w := maxInt / 4
	return image.Rect(0, 0, w, 5)
}

func (f *hugeBoundsFilter) Draw(dst draw.Image, src image.Image, options *Options) {}

func TestPlanCustomFilterStillSupported(t *testing.T) {
	// External-style custom filters implementing Filter must work unchanged.
	src := newProceduralImage(image.Pt(0, 0), 6, 5)
	custom := &testFilter{z: 2}
	g := New(custom, Grayscale())
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		t.Fatal(err)
	}
	ws, err := NewWorkspace(math.MaxInt64)
	if err != nil {
		t.Fatal(err)
	}
	dst := image.NewNRGBA64(plan.OutputBounds())
	if err := plan.Draw(ws, dst, src); err != nil {
		t.Fatal(err)
	}
	oldDst := image.NewNRGBA64(g.Bounds(src.Bounds()))
	g.Draw(oldDst, src)
	assertPixelsEqual(t, dst, oldDst, "custom filter")
}

func BenchmarkPlanFirstVsHot(b *testing.B) {
	src := newProceduralImage(image.Pt(0, 0), 200, 150)
	g := New(
		Resize(240, 0, LanczosResampling),
		GaussianBlur(1.5),
		Grayscale(),
		Rotate90(),
	)
	plan, err := g.Compile(src.Bounds())
	if err != nil {
		b.Fatal(err)
	}
	ws, err := NewWorkspace(math.MaxInt64)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("old_path", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			dst := image.NewNRGBA64(g.Bounds(src.Bounds()))
			g.Draw(dst, src)
		}
	})
	b.Run("plan_first_call", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			local, _ := NewWorkspace(math.MaxInt64)
			dst := image.NewNRGBA64(plan.OutputBounds())
			if err := plan.Draw(local, dst, src); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("plan_hot_reuse", func(b *testing.B) {
		b.ReportAllocs()
		dst := image.NewNRGBA64(plan.OutputBounds())
		// Warm the pool once so steady-state reuse is measured.
		if err := plan.Draw(ws, dst, src); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := plan.Draw(ws, dst, src); err != nil {
				b.Fatal(err)
			}
		}
	})
}
