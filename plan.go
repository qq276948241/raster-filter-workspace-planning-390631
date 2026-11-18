package gift

import (
	"errors"
	"image"
	"image/draw"
	"math"
	"math/bits"
	sortpkg "sort"
	"sync"
)

// Errors returned by the plan compilation and execution functions.
var (
	// ErrInvalidArgument is returned when a mandatory argument (workspace, source or
	// destination image) is missing.
	ErrInvalidArgument = errors.New("gift: invalid argument")

	// ErrInvalidBounds is returned when the supplied bounds are not canonical
	// (negative width or height) or a filter reports such bounds during compilation.
	ErrInvalidBounds = errors.New("gift: invalid image bounds")

	// ErrOverflow is returned when an intermediate image width/height or pixel
	// count does not fit into the platform int range.
	ErrOverflow = errors.New("gift: image size overflow")

	// ErrSourceSizeMismatch is returned when the source image size does not match
	// the size the plan was compiled for.
	ErrSourceSizeMismatch = errors.New("gift: source image size does not match the plan")

	// ErrDstTooSmall is returned when the destination image cannot hold the whole
	// planned output. No destination pixel is modified in this case.
	ErrDstTooSmall = errors.New("gift: destination image bounds are too small")

	// ErrBudgetExceeded is returned when the workspace pixel budget is lower than
	// the number of intermediate pixels that must be alive at the same time.
	ErrBudgetExceeded = errors.New("gift: workspace pixel budget exceeded")

	// ErrWorkspaceBusy is returned when the same workspace is used concurrently
	// by two executions. A workspace may only serve one execution at a time.
	ErrWorkspaceBusy = errors.New("gift: workspace is already in use")
)

// Platform-dependent int range, used by the checked arithmetic helpers.
const (
	maxInt = int64(^uint(0) >> 1)
	minInt = -maxInt - 1
)

// addInt returns x+y and ok=false if the result overflows the int range.
func addInt(x, y int) (int, bool) {
	sum, _ := bits.Add64(uint64(x), uint64(y), 0)
	// Signed overflow: operands share a sign bit but the result does not.
	if (^(uint64(x) ^ uint64(y)) & (uint64(x) ^ sum) & (1 << 63)) != 0 {
		return 0, false
	}
	s := int64(sum)
	if s > maxInt || s < minInt {
		return 0, false
	}
	return int(s), true
}

// rectPixelCount returns the number of pixels in r. It fails on non-canonical
// rectangles and when the width*height product overflows the platform int
// range.
func rectPixelCount(r image.Rectangle) (int64, error) {
	dx, dy := int64(r.Dx()), int64(r.Dy())
	if dx < 0 || dy < 0 {
		return 0, ErrInvalidBounds
	}
	hi, lo := bits.Mul64(uint64(dx), uint64(dy))
	if hi != 0 || lo > math.MaxInt64 || int64(lo) > maxInt {
		return 0, ErrOverflow
	}
	return int64(lo), nil
}

// addRectOrigin translates r by min, failing when the resulting coordinates
// overflow the int range.
func addRectOrigin(r image.Rectangle, min image.Point) (image.Rectangle, error) {
	minX, ok1 := addInt(min.X, r.Min.X)
	minY, ok2 := addInt(min.Y, r.Min.Y)
	maxX, ok3 := addInt(min.X, r.Max.X)
	maxY, ok4 := addInt(min.Y, r.Max.Y)
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return image.Rectangle{}, ErrOverflow
	}
	return image.Rect(minX, minY, maxX, maxY), nil
}

// planStage is a single compiled filter step. Both bounds rectangles are
// snapshots: later changes to the original filter list cannot affect them.
type planStage struct {
	filter    Filter
	inBounds  image.Rectangle
	outBounds image.Rectangle
}

// Plan is an immutable, reusable execution plan compiled from a filter list.
//
// A plan holds a snapshot of the filter list, a snapshot of the processing
// options, the input/output bounds of every filter stage and the precomputed
// number of intermediate pixels required for an execution. Plans are safe for
// concurrent use: any number of goroutines may execute the same plan, each one
// with its own Workspace. Modifying the original GIFT filter list or toggling
// parallelization after compilation has no effect on an already compiled plan.
//
// A plan is bound to the source image size it was compiled for. Source images
// may have a non-zero or negative origin; only the width and height are part of
// the binding.
type Plan struct {
	filters   []Filter
	options   Options
	srcSize   image.Point
	outBounds image.Rectangle
	stages    []planStage

	// tempPeak is the maximum number of intermediate (workspace-owned) pixels
	// alive at the same time during a Draw execution.
	tempPeak int64
	// finalPixels is the area of the final output image.
	finalPixels int64
}

// Compile builds a reusable execution plan for source images having the given
// bounds. Only the source width and height matter; a non-zero or negative
// origin is supported. An empty (zero-area) source is allowed and produces a
// plan that performs no writes.
//
// The filter list and the current options are snapshotted: subsequent calls of
// Add, Empty or SetParallelization on the GIFT value do not affect the plan.
func (g *GIFT) Compile(srcBounds image.Rectangle) (*Plan, error) {
	if srcBounds.Dx() < 0 || srcBounds.Dy() < 0 {
		return nil, ErrInvalidBounds
	}

	p := &Plan{
		filters: append([]Filter(nil), g.Filters...),
		options: g.Options,
		srcSize: srcBounds.Size(),
	}

	b := srcBounds
	var live int64
	for _, f := range p.filters {
		in := b
		out := f.Bounds(in)
		area, err := rectPixelCount(out)
		if err != nil {
			return nil, err
		}
		p.stages = append(p.stages, planStage{
			filter:    f,
			inBounds:  in,
			outBounds: out,
		})
		b = out

		// Every stage output except the last one is a workspace-owned temp.
		// The temp of the previous stage is released once the next stage has
		// been drawn, so at most two temps are alive at the same time.
		if len(p.stages) < len(p.filters) {
			live += area
			if live > p.tempPeak {
				p.tempPeak = live
			}
		}
		if len(p.stages) > 1 {
			if prev := p.stages[len(p.stages)-2].outBounds; prev.Dx() > 0 && prev.Dy() > 0 {
				if prevArea, err := rectPixelCount(prev); err == nil {
					live -= prevArea
				}
			}
		}
	}

	p.outBounds = b
	area, err := rectPixelCount(b)
	if err != nil {
		return nil, err
	}
	p.finalPixels = area
	return p, nil
}

// StageBounds returns the snapshotted input and output bounds of the filter
// stage at index i. The returned rectangles are copies; mutating them cannot
// affect the plan. It panics when i is out of range.
func (p *Plan) StageBounds(i int) (inBounds, outBounds image.Rectangle) {
	st := p.stages[i]
	return st.inBounds, st.outBounds
}

// Len returns the number of compiled filter stages.
func (p *Plan) Len() int {
	return len(p.stages)
}

// OutputBounds returns the bounds of the image produced by the plan. They are
// equivalent to the bounds returned by GIFT.Bounds at compilation time.
func (p *Plan) OutputBounds() image.Rectangle {
	return p.outBounds
}

// SourceSize returns the source image width and height the plan was compiled
// for.
func (p *Plan) SourceSize() image.Point {
	return p.srcSize
}

// Options returns the snapshotted processing options.
func (p *Plan) Options() Options {
	return p.options
}

// RequiredPixels returns the maximum number of simultaneously alive
// intermediate pixels required by Draw. The source and destination images are
// owned by the caller and are never counted against the workspace budget.
func (p *Plan) RequiredPixels() int64 {
	return p.tempPeak
}

// RequiredDrawAtPixels returns the intermediate pixel budget required by
// DrawAt. DrawAt needs an additional output-sized temp unless the result can be
// drawn straight into the destination (CopyOperator at the destination origin
// or into a sub-image).
func (p *Plan) RequiredDrawAtPixels() (int64, error) {
	sum, carry := bits.Add64(uint64(p.tempPeak), uint64(p.finalPixels), 0)
	if carry != 0 || sum > math.MaxInt64 {
		return 0, ErrOverflow
	}
	return int64(sum), nil
}

// tempLease pairs a temp image handed to a filter with the (possibly larger)
// pooled buffer backing it.
type tempLease struct {
	view *image.NRGBA64
	buf  *pooledBuffer
}

// pooledBuffer is a reusable zero-origin NRGBA64 backing array.
type pooledBuffer struct {
	img       *image.NRGBA64
	capPixels int64
}

// fits reports whether the backing array can serve a rectangle of the given
// relative size. The pixel capacity alone is insufficient: a square buffer may
// hold enough pixels while being too narrow for a wide rectangle's stride.
func (b *pooledBuffer) fits(w, h int) bool {
	return b.img.Rect.Dx() >= w && b.img.Rect.Dy() >= h
}

// view returns an NRGBA64 sharing the backing array whose relative rectangle
// is (0,0)-(w,h) translated to r.Min. Keeping the relative origin at zero is
// required because the pixel stride belongs to the parent buffer.
func (b *pooledBuffer) view(r image.Rectangle) *image.NRGBA64 {
	view := *b.img
	view.Rect = image.Rectangle{
		Min: r.Min,
		Max: r.Min.Add(image.Pt(r.Dx(), r.Dy())),
	}
	return &view
}

// Workspace carries the temporary images and the resource budget used by plan
// executions. A workspace is reusable across sequential executions: buffers
// allocated for the first image are handed out again for the next one, so a
// hot execution performs no large pixel allocations.
//
// A workspace must not be shared by concurrent executions; concurrent use of
// the same workspace returns ErrWorkspaceBusy. Different workspaces may be used
// concurrently, even with the same plan. There is no global cache: all retained
// memory belongs to the workspace and is released by Release or when the
// workspace becomes unreachable.
type Workspace struct {
	mu sync.Mutex

	budget int64
	busy   bool
	live   int64

	// Retained buffers, sorted by capacity ascending. At most two buffers are
	// retained (the widest pipeline stage needs two simultaneous temps), and
	// buffers larger than the biggest temp required by the most recent run are
	// dropped instead of being kept forever.
	pool []*pooledBuffer

	// allocations counts the backing pixel arrays created since the workspace
	// was created; it is used by tests to demonstrate hot reuse.
	allocations int64
}

// NewWorkspace creates a workspace with the given intermediate pixel budget.
// The budget counts pixels of workspace-owned temps that are alive at the same
// time; it never includes the caller-owned source or destination images. A
// negative budget is rejected up front. A zero budget is valid for plans and
// executions that need no temps.
func NewWorkspace(budgetPixels int64) (*Workspace, error) {
	if budgetPixels < 0 {
		return nil, ErrBudgetExceeded
	}
	return &Workspace{budget: budgetPixels}, nil
}

// Budget returns the workspace pixel budget.
func (ws *Workspace) Budget() int64 {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	return ws.budget
}

// RetainedPixels returns the total pixel capacity currently kept in the
// workspace buffer pool.
func (ws *Workspace) RetainedPixels() int64 {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	var total int64
	for _, b := range ws.pool {
		total += b.capPixels
	}
	return total
}

// Release frees all retained temporary image memory. It must not be called
// while an execution is in progress. After Release the workspace can be used
// again; the next execution allocates fresh buffers.
func (ws *Workspace) Release() {
	ws.mu.Lock()
	ws.pool = nil
	ws.live = 0
	ws.busy = false
	ws.mu.Unlock()
}

// begin claims the workspace for one execution. It fails fast on concurrent
// misuse instead of corrupting the buffer pool.
func (ws *Workspace) begin() error {
	ws.mu.Lock()
	if ws.busy {
		ws.mu.Unlock()
		return ErrWorkspaceBusy
	}
	ws.busy = true
	ws.mu.Unlock()
	return nil
}

// acquire returns a temp image with the given bounds. A pooled backing array is
// reused when one is large enough; otherwise a new array is allocated. The
// returned image shares the backing array and may carry an arbitrary origin.
func (ws *Workspace) acquire(r image.Rectangle) (tempLease, error) {
	if r.Dx() <= 0 || r.Dy() <= 0 {
		// Empty temps allocate nothing and never enter the pool.
		return tempLease{view: image.NewNRGBA64(r)}, nil
	}
	area, err := rectPixelCount(r)
	if err != nil {
		return tempLease{}, err
	}

	ws.mu.Lock()
	defer ws.mu.Unlock()

	if ws.live+area < ws.live || ws.live+area > ws.budget {
		return tempLease{}, ErrBudgetExceeded
	}

	// Best fit: the smallest retained buffer whose backing array covers the
	// requested rectangle. Matching on both dimensions (not just the pixel
	// count) prevents a square buffer from serving a too-wide rectangle. The
	// pool is tiny (at most two temps are ever alive simultaneously), so a
	// linear scan is appropriate.
	idx := -1
	for i, b := range ws.pool {
		if b.fits(r.Dx(), r.Dy()) && (idx == -1 || b.capPixels < ws.pool[idx].capPixels) {
			idx = i
		}
	}
	if idx >= 0 {
		buf := ws.pool[idx]
		ws.pool = append(ws.pool[:idx], ws.pool[idx+1:]...)
		ws.live += area
		return tempLease{view: buf.view(r), buf: buf}, nil
	}

	ws.live += area
	ws.allocations++
	parent := image.NewNRGBA64(image.Rect(0, 0, r.Dx(), r.Dy()))
	buf := &pooledBuffer{
		img:       parent,
		capPixels: area,
	}
	return tempLease{view: buf.view(r), buf: buf}, nil
}

// finish returns the leased buffers to the pool, resets the live accounting and
// releases the busy flag. Buffers larger than the biggest temp used in this run
// are dropped so that a one-off large image cannot pin a large buffer forever,
// and the retained total never exceeds the budget.
func (ws *Workspace) finish(leases []tempLease, runMaxArea int64) {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	kept := make([]*pooledBuffer, 0, len(ws.pool)+len(leases))
	kept = append(kept, ws.pool...)
	for _, l := range leases {
		if l.buf != nil {
			kept = append(kept, l.buf)
		}
	}

	// Drop every buffer larger than this run's biggest temp: a one-off large
	// image must not pin a large buffer once the workload shrinks again.
	filtered := kept[:0]
	for _, b := range kept {
		if b.capPixels <= runMaxArea {
			filtered = append(filtered, b)
		}
	}
	kept = filtered

	sortpkg.Slice(kept, func(i, j int) bool {
		return kept[i].capPixels < kept[j].capPixels
	})

	// Never retain more pixel capacity than the budget allows.
	var total int64
	for i := range kept {
		b := kept[i]
		if total+b.capPixels > ws.budget {
			kept = kept[:i]
			break
		}
		total += b.capPixels
	}
	ws.pool = kept

	ws.live = 0
	ws.busy = false
}

// validateExecution performs every up-front check before the destination image
// is touched.
func (p *Plan) validateExecution(ws *Workspace, dst draw.Image, src image.Image, required int64) error {
	if ws == nil || dst == nil || src == nil {
		return ErrInvalidArgument
	}
	sb := src.Bounds()
	if sb.Dx() < 0 || sb.Dy() < 0 {
		return ErrInvalidBounds
	}
	if sb.Size() != p.srcSize {
		return ErrSourceSizeMismatch
	}
	if p.finalPixels > 0 {
		// The result is written starting at the destination Min point, so only
		// the output size is tested against the destination rectangle.
		need := image.Rectangle{
			Min: dst.Bounds().Min,
			Max: dst.Bounds().Min.Add(p.outBounds.Size()),
		}
		if !need.In(dst.Bounds()) {
			return ErrDstTooSmall
		}
	}
	if required > ws.budget {
		return ErrBudgetExceeded
	}
	return nil
}

// render runs the compiled filter chain into dst. The caller must have claimed
// the workspace and is responsible for finishing it. Acquired temps are
// appended to leases and the largest requested area is tracked in runMaxArea.
func (p *Plan) render(ws *Workspace, dst draw.Image, src image.Image, leases *[]tempLease, runMaxArea *int64) error {
	if len(p.stages) == 0 {
		copyimage(dst, src, &p.options)
		return nil
	}

	last := len(p.stages) - 1
	var tmpIn image.Image = src
	for i, st := range p.stages {
		var tmpOut draw.Image
		if i == last {
			tmpOut = dst
		} else {
			lease, err := ws.acquire(st.outBounds)
			if err != nil {
				return err
			}
			if area, aerr := rectPixelCount(st.outBounds); aerr == nil && area > *runMaxArea {
				*runMaxArea = area
			}
			*leases = append(*leases, lease)
			tmpOut = lease.view
		}
		st.filter.Draw(tmpOut, tmpIn, &p.options)
		tmpIn = tmpOut
	}
	return nil
}

// Draw applies the compiled filter chain to src and writes the result into dst.
// All validation (source size, destination writable range, pixel budget)
// happens before any destination pixel is written; a failed validation leaves
// the destination pixels untouched.
//
// The produced pixels are identical to GIFT.Draw with the same filter list and
// options.
func (p *Plan) Draw(ws *Workspace, dst draw.Image, src image.Image) error {
	if err := p.validateExecution(ws, dst, src, p.tempPeak); err != nil {
		return err
	}
	if err := ws.begin(); err != nil {
		return err
	}
	leases := make([]tempLease, 0, len(p.stages))
	var runMaxArea int64
	defer func() { ws.finish(leases, runMaxArea) }()

	return p.render(ws, dst, src, &leases, &runMaxArea)
}

// DrawAt applies the compiled filter chain and places the result into dst at
// point pt using the composition operator op, mirroring GIFT.DrawAt.
//
// When a temporary image is needed, it is allocated from the workspace; the
// required budget is RequiredDrawAtPixels instead of RequiredPixels. Pixels
// landing outside the destination bounds are clipped and never written, and a
// fully clipped placement performs no writes.
func (p *Plan) DrawAt(ws *Workspace, dst draw.Image, src image.Image, pt image.Point, op Operator) error {
	if ws == nil || dst == nil || src == nil {
		return ErrInvalidArgument
	}
	sb := src.Bounds()
	if sb.Dx() < 0 || sb.Dy() < 0 {
		return ErrInvalidBounds
	}
	if sb.Size() != p.srcSize {
		return ErrSourceSizeMismatch
	}

	if op == CopyOperator {
		// Fast paths identical to GIFT.DrawAt: no placement temp is needed.
		if pt.Eq(dst.Bounds().Min) {
			return p.Draw(ws, dst, src)
		}
		if subimg, ok := getSubImage(dst, pt); ok {
			if err := p.validateExecution(ws, subimg, src, p.tempPeak); err != nil {
				return err
			}
			if err := ws.begin(); err != nil {
				return err
			}
			leases := make([]tempLease, 0, len(p.stages))
			var runMaxArea int64
			defer func() { ws.finish(leases, runMaxArea) }()
			return p.render(ws, subimg, src, &leases, &runMaxArea)
		}
	}

	// The placement needs a full output-sized temp.
	tb, err := addRectOrigin(p.outBounds.Sub(p.outBounds.Min), pt)
	if err != nil {
		return err
	}
	ib := tb.Intersect(dst.Bounds())
	if ib.Empty() {
		// Nothing lands inside the destination: deterministic no-op, no budget
		// is consumed and no destination pixel is touched.
		return nil
	}

	required, err := p.RequiredDrawAtPixels()
	if err != nil {
		return err
	}
	if required > ws.budget {
		return ErrBudgetExceeded
	}
	if err := ws.begin(); err != nil {
		return err
	}
	leases := make([]tempLease, 0, len(p.stages)+1)
	var runMaxArea int64
	defer func() { ws.finish(leases, runMaxArea) }()

	lease, err := ws.acquire(tb)
	if err != nil {
		return err
	}
	leases = append(leases, lease)
	if p.finalPixels > runMaxArea {
		runMaxArea = p.finalPixels
	}
	if err := p.render(ws, lease.view, src, &leases, &runMaxArea); err != nil {
		return err
	}

	switch op {
	case OverOperator:
		composeOver(dst, lease.view, tb, &p.options)
	default:
		composeCopy(dst, lease.view, tb, &p.options)
	}
	return nil
}
