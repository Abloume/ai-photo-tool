package analyzer

import "testing"

func TestEstimateCCTWarmCool(t *testing.T) {
	// 偏红（暖色）图 → 相关色温应该偏低（< 5000 判暖）
	warm := estimateCCT(200, 120, 100)
	if warm > 5500 {
		t.Errorf("warm image CCT = %v, want low (<5500)", warm)
	}
	// 偏蓝（冷色）图 → 色温应该偏高（> 6500 判冷）
	cool := estimateCCT(100, 140, 210)
	if cool < 6000 {
		t.Errorf("cool image CCT = %v, want high (>6000)", cool)
	}
	// 纯灰（R=G=B）应落在中性区间
	neutral := estimateCCT(128, 128, 128)
	if neutral < 4000 || neutral > 9000 {
		t.Errorf("neutral gray CCT = %v, want 4000~9000", neutral)
	}
}
