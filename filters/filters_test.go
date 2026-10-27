package filters

import "testing"

func TestClamp255(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{-10, 0},
		{0, 0},
		{128, 128},
		{255, 255},
		{300, 255},
	}
	for _, c := range cases {
		if got := clamp255(c.in); got != c.want {
			t.Errorf("clamp255(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestAdjustTemperature(t *testing.T) {
	r, g, b := 100.0, 100.0, 100.0
	// 暖调（t=+100）：红+30，蓝-30
	wr, _, wb := adjustTemperature(r, g, b, 100)
	if wr != 130 || wb != 70 {
		t.Errorf("warm: got r=%v b=%v, want r=130 b=70", wr, wb)
	}
	// 冷调（t=-100）：红-30，蓝+30
	cr, _, cb := adjustTemperature(r, g, b, -100)
	if cr != 70 || cb != 130 {
		t.Errorf("cool: got r=%v b=%v, want r=70 b=130", cr, cb)
	}
	// t=0 不变
	if nr, ng, nb := adjustTemperature(r, g, b, 0); nr != 100 || ng != 100 || nb != 100 {
		t.Errorf("t=0 should be identity, got %v,%v,%v", nr, ng, nb)
	}
}

func TestAdjustSaturationIdentity(t *testing.T) {
	// s=100 时 scale=1，结果应等于原图
	r, g, b := 80.0, 160.0, 200.0
	nr, ng, nb := adjustSaturation(r, g, b, 100)
	if nr != r || ng != g || nb != b {
		t.Errorf("saturation=100 should be identity, got %v,%v,%v", nr, ng, nb)
	}
}

func TestHashNoiseDeterministic(t *testing.T) {
	// 同一像素两次结果必须一致（并行/重算可复现的基础）
	a := hashNoise(120, 80)
	b := hashNoise(120, 80)
	if a != b {
		t.Errorf("hashNoise not deterministic: %v != %v", a, b)
	}
	// 范围在 [-1, 1)
	if a < -1.0 || a >= 1.0 {
		t.Errorf("hashNoise out of range: %v", a)
	}
	// 不同像素应当给出不同噪声值（抽样几十个坐标，全相同就说明哈希坏了）
	seen := map[float64]bool{}
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			seen[hashNoise(x, y)] = true
		}
	}
	if len(seen) < 300 {
		t.Errorf("hashNoise too few distinct values: %d/400", len(seen))
	}
}
