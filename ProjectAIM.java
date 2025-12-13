import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public final class ProjectAIM implements HeuristicComponent {

    private final AimHeuristicCheck parent;
    private final List<Float> yawSamples = new ArrayList<>(120);
    private final List<Float> pitchSamples = new ArrayList<>(120);
    private final List<Double> rankHistory = new ArrayList<>(12);

    private float prevYaw, prevPitch;
    private float bufRank, bufLinear, bufPattern, bufGcd;
    private final List<Long> gcdHistory = new ArrayList<>(32);
    private int lowGcdCount;

    public ProjectAIM(AimHeuristicCheck parent) {
        this.parent = parent;
    }

    @Override
    public void process(RotationEvent e) {
        Vec2f abs = e.getAbsDelta();
        float dy = abs.getX();
        float dp = abs.getY();

        if (dy < 0.001f && dp < 0.001f) return;

        yawSamples.add(e.getDelta().getX());
        pitchSamples.add(e.getDelta().getY());

        if (yawSamples.size() >= 100) {
            runAnalysis();
            yawSamples.clear();
            pitchSamples.clear();
        }

        if (dy > 1.35f || (dp > 1.35f && dy > 0.32f)) {
            runGcdCheck(dy, dp);
        }

        tick();
        prevYaw = dy;
        prevPitch = dp;
    }

    private void runAnalysis() {
        PlayerProfile p = parent.getProfile();

        List<Float> jY = jiff(yawSamples, 5);
        List<Float> jP = jiff(pitchSamples, 5);
        if (jY.isEmpty() || jP.isEmpty()) return;

        List<Float> ratios = new ArrayList<>();
        int len = Math.min(jY.size(), jP.size());
        int infCount = 0;

        for (int i = 0; i < len; i++) {
            float py = jP.get(i);
            float jy = jY.get(i);
            if (py != 0) {
                ratios.add(jy / py);
            } else if (jy != 0) {
                ratios.add(Float.POSITIVE_INFINITY);
                infCount++;
            }
        }

        if (!ratios.isEmpty()) {
            double iqr = iqr(ratios);
            boolean suspicious = iqr > 12.5 && iqr < 96 && infCount > 0;

            if (suspicious) {
                bufPattern += iqr > 20 ? 1.4f : 0.8f;
                if (bufPattern > 11f) {
                    p.punish("Aim", "ProjectAIM", "iqr=" + fmt(iqr) + " inf=" + infCount, 2.5f);
                    bufPattern = 9f;
                }
            } else {
                bufPattern = dec(bufPattern, iqr < 7 ? 5f : 3.5f);
            }
        }

        double avgY = Statistics.getAverage(yawSamples);
        int patterns = countJiffPatterns(jY);

        if (patterns > 2 && avgY > 3.0 && patterns != 4 && patterns != 6 && patterns != 12) {
            p.punish("Aim", "ProjectAIM", "bot p=" + patterns, 3.5f);
        }

        List<Double> devs = new ArrayList<>();
        List<Float> chunk = new ArrayList<>();
        int distinctSum = 0;

        for (float yaw : yawSamples) {
            chunk.add(yaw);
            if (chunk.size() >= 10) {
                distinctSum += distinct(jiff(chunk, 4));
                devs.add(Statistics.getStandardDeviation(jiff(chunk, 5)));
                chunk.clear();
            }
        }

        List<Double> outliers = zOutliers(devs, 0.5);
        boolean tooLinear = outliers.isEmpty() || (outliers.size() == 1 && Math.abs(outliers.get(0)) > 10 && Math.abs(outliers.get(0)) < 100);

        if (tooLinear) {
            bufLinear += 1.5f;
            if (bufLinear > 8f) {
                p.punish("Aim", "ProjectAIM", "linear out=" + outliers.size(), 3.5f);
                bufLinear = 5f;
            }
        } else {
            bufLinear = dec(bufLinear, 1f);
        }

        float rank = distinctSum / 60f;
        rankHistory.add((double) rank);

        if (rankHistory.size() >= 10) {
            double avg = Statistics.getAverage(rankHistory);
            int normal = 0;
            for (double d : rankHistory) if (d > 0.97) normal++;

            if (avg < 0.95 && normal < 4) {
                p.punish("Aim", "ProjectAIM", "longterm avg=" + fmt(avg) + " n=" + normal, 5.5f);
            }
            rankHistory.clear();
        }

        if (rank > 0.7f && rank < 1f && Math.abs(avgY) > 1.8) {
            float inc = rank > 0.9f ? 0.08f : rank > 0.8f ? 2f : 3f;
            if (bufRank < 0.01f && rank < 0.8f) {
                bufRank += 0.2f;
            } else if (bufRank >= 0.01f) {
                bufRank += inc;
                if (bufRank > 6f) {
                    p.punish("Aim", "ProjectAIM", "rank=" + fmt(rank), 2f);
                    bufRank = 5f;
                }
            }
        } else {
            bufRank = dec(bufRank, 2.25f);
        }
    }

    private void runGcdCheck(float dy, float dp) {
        if (dy < 0.25f || dp < 0.25f || dy > 25f || dp > 25f) return;
        if (prevPitch < 0.25f) return;

        long exp = (long) (Statistics.EXPANDER * dp);
        long expPrev = (long) (Statistics.EXPANDER * prevPitch);
        long gcd = gcd(exp, expPrev);

        if (gcd > 0) {
            gcdHistory.add(gcd);
            if (gcd < 120000L) lowGcdCount++;
        }

        if (gcdHistory.size() >= 20) {
            int total = gcdHistory.size();
            int highCount = 0;
            for (long g : gcdHistory) {
                if (g > 250000L) highCount++;
            }

            double lowRatio = (double) lowGcdCount / total;
            double highRatio = (double) highCount / total;

            List<Long> lowOnly = new ArrayList<>();
            for (long g : gcdHistory) {
                if (g < 150000L) lowOnly.add(g);
            }

            int uniqueLow = new HashSet<>(lowOnly).size();
            double uniqueRatio = lowOnly.isEmpty() ? 0 : (double) uniqueLow / lowOnly.size();

            boolean mostlyLow = lowRatio > 0.75;
            boolean noHighSpikes = highRatio < 0.1;
            boolean chaotic = uniqueRatio > 0.65;

            if (mostlyLow && noHighSpikes && chaotic) {
                bufGcd += 3.5f;
                if (bufGcd > 5f) {
                    parent.getProfile().punish("Aim", "ProjectAIM", "gcd low=" + fmt(lowRatio) + " high=" + highCount, 3f);
                    bufGcd = 2f;
                }
            } else if (highRatio > 0.15) {
                bufGcd = dec(bufGcd, 4f);
            } else {
                bufGcd = dec(bufGcd, 1f);
            }

            gcdHistory.clear();
            lowGcdCount = 0;
        }
    }

    private void tick() {
        bufRank = dec(bufRank, 0.05f);
        bufLinear = dec(bufLinear, 0.08f);
        bufPattern = dec(bufPattern, 0.1f);
        bufGcd = dec(bufGcd, 0.08f);
    }

    private List<Float> jiff(List<Float> src, int depth) {
        List<Float> r = new ArrayList<>(src);
        for (int d = 0; d < depth; d++) {
            List<Float> next = new ArrayList<>();
            for (int i = 1; i < r.size(); i++) {
                next.add(Math.abs(Math.abs(r.get(i)) - Math.abs(r.get(i - 1))));
            }
            r = next;
        }
        return r;
    }

    private int distinct(List<Float> data) {
        return new HashSet<>(data).size();
    }

    private double iqr(List<Float> data) {
        if (data.size() < 4) return 0;
        List<Float> s = new ArrayList<>(data);
        s.sort(Float::compareTo);
        return s.get(3 * s.size() / 4) - s.get(s.size() / 4);
    }

    private List<Double> zOutliers(List<Double> data, double th) {
        if (data.size() < 3) return new ArrayList<>();
        double mean = Statistics.getAverage(data);
        double std = Statistics.getStandardDeviation(data);
        if (std < 1e-6) return new ArrayList<>();

        List<Double> out = new ArrayList<>();
        for (double v : data) {
            if (Math.abs((v - mean) / std) > th) out.add(v);
        }
        return out;
    }

    private int countJiffPatterns(List<Float> jiff) {
        int c = 0;
        for (int i = 0; i < jiff.size(); i++) {
            float f = jiff.get(i);
            if (f == 0 || !String.valueOf(f).contains("E")) continue;
            for (int j = i + 1; j < jiff.size(); j++) {
                if (f == jiff.get(j)) c++;
            }
        }
        return c;
    }

    private long gcd(long a, long b) {
        return b <= 16384L ? a : gcd(b, a % b);
    }

    private float dec(float v, float by) {
        return Math.max(0, v - by);
    }

    private String fmt(double v) {
        return String.format("%.2f", v);
    }
}
