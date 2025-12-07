
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


public final class ProjectAIM implements HeuristicComponent {
    private final AimHeuristicCheck check;
    private final List<Float> yawDeltas = new ArrayList<>();
    private final List<Float> pitchDeltas = new ArrayList<>();
    private final List<Double> distinctRankHistory = new ArrayList<>();
    private float lastDy = 0.0f;
    private float lastDp = 0.0f;
    private float rankBuffer = 0.0f;
    private float linearBuffer = 0.0f;
    private float patternBuffer = 0.0f;
    private int sampleCount = 0;
    private static final int SAMPLE_SIZE = 100;
    private static final double EXPANDER = Math.pow(2, 24);
    public ProjectAIM(AimHeuristicCheck check) {
        this.check = check;
    }


    @Override
    public void process(RotationEvent event) {
        Vec2f delta = event.getDelta();
        Vec2f absDelta = event.getAbsDelta();

        float dy = absDelta.getX();
        float dp = absDelta.getY();

        if (dy == 0 && dp == 0) {
            return;
        }
        sampleCount++;
        yawDeltas.add(delta.getX());
        pitchDeltas.add(delta.getY());
        if (yawDeltas.size() >= SAMPLE_SIZE) {
            analyzeRotations();
            yawDeltas.clear();
            pitchDeltas.clear();
        }


        if (dy > 1.35 || (dp > 1.35 && dy > 0.32)) {
            checkLimitedRotation(dy, dp);
        }
        decayBuffers();

        lastDy = dy;
        lastDp = dp;
    }
    private void analyzeRotations() {
        PlayerProfile profile = check.getProfile();
        List<Float> jiffYaw = getJiffDelta(yawDeltas, 5);
        List<Float> jiffPitch = getJiffDelta(pitchDeltas, 5);
        if (jiffYaw.isEmpty() || jiffPitch.isEmpty()) return;
        List<Float> jiffOmni = new ArrayList<>();
        int len = Math.min(jiffYaw.size(), jiffPitch.size());

        for (int i = 0; i < len; i++) {
            float py = jiffPitch.get(i);
            float jy = jiffYaw.get(i);
            if (py != 0) {
                jiffOmni.add(jy / py);
            } else if (jy != 0) {
                jiffOmni.add(Float.POSITIVE_INFINITY);
            }
        }
        if (!jiffOmni.isEmpty()) {
            int infs = 0;
            for (float j : jiffOmni) {
                if (Float.isInfinite(j)) infs++;
            }
            double iqr = getIQR(jiffOmni);

            if (iqr > 12.5 && iqr < 96 && infs > 0) {
                patternBuffer += (iqr > 20) ? 1.4f : 0.8f;

                if (patternBuffer > 11.0f) {
                    profile.punish("Aim", "ProjectAIM",
                            String.format("IQR iqr=%.2f infs=%d", iqr, infs), 2.5f);
                    patternBuffer = 9.0f;
                }
            } else if (iqr < 13 || infs == 0) {
                patternBuffer = Math.max(0, patternBuffer - ((iqr < 7) ? 5.0f : 3.5f));
            }
        }
        int jiffPatterns = 0;
        double avgYaw = Statistics.getAverage(yawDeltas);
        for (int i = 0; i < jiffYaw.size(); i++) {
            float f = jiffYaw.get(i);
            if (!String.valueOf(f).contains("E") || f == 0) continue;

            for (int r = 0; r < jiffYaw.size(); r++) {
                if (r == i) continue;
                if (f == jiffYaw.get(r)) {
                    jiffPatterns++;
                }
            }
        }
        if (jiffPatterns > 2 && avgYaw > 3.0
                && jiffPatterns != 6 && jiffPatterns != 12 && jiffPatterns != 4) {
            profile.punish("Aim", "ProjectAIM",
                    String.format("BotPattern p=%d", jiffPatterns), 3.5f);
        }


        List<Float> jiffForKS = getJiffDelta(yawDeltas, 6);
        List<Double> deviations = new ArrayList<>();
        List<Float> yawStack = new ArrayList<>();
        int resultDistinct = 0;
        for (float yaw : yawDeltas) {
            yawStack.add(yaw);
            if (yawStack.size() >= 10) {
                List<Float> jiff = getJiffDelta(yawStack, 4);
                resultDistinct += getDistinct(jiff);
                deviations.add(Statistics.getStandardDeviation(getJiffDelta(yawStack, 5)));
                yawStack.clear();
            }
        }
        List<Double> outliers = getZScoreOutliers(deviations, 0.5);

        if (outliers.isEmpty() || (outliers.size() == 1 && Math.abs(outliers.get(0)) > 10 && Math.abs(outliers.get(0)) < 100)) {
            linearBuffer += 1.5f;

            if (linearBuffer > 8.0f) {
                profile.punish("Aim", "ProjectAIM",
                        String.format("Linear out=%d", outliers.size()), 3.5f);
                linearBuffer = 5.0f;
            }
        } else {
            linearBuffer = Math.max(0, linearBuffer - 1.0f);
        }
        float distinctRank = (float) resultDistinct / 60.0f;
        distinctRankHistory.add((double) distinctRank);

        if (distinctRankHistory.size() >= 10) {
            double avgRank = Statistics.getAverage(distinctRankHistory);
            int normalCount = 0;

            for (double d : distinctRankHistory) {
                if (d > 0.97) normalCount++;
            }

            if (avgRank < 0.95 && normalCount < 4) {
                profile.punish("Aim", "ProjectAIM",
                        String.format("LongTerm avg=%.3f norm=%d", avgRank, normalCount), 5.5f);
            }
            distinctRankHistory.clear();
        }
        if (distinctRank < 1.0 && distinctRank > 0.7 && Math.abs(avgYaw) > 1.8) {
            float increment = (distinctRank > 0.9) ? 0.08f : (distinctRank > 0.8) ? 2.0f : 3.0f;

            if (rankBuffer < 0.01) {
                if (distinctRank < 0.8) {
                    rankBuffer += 0.2f;
                }
            } else {
                rankBuffer += increment;

                if (rankBuffer > 6.0f) {
                    profile.punish("Aim", "ProjectAIM",
                            String.format("Rank r=%.3f", distinctRank), 2.0f);
                    rankBuffer = 5.0f;
                }
            }
        } else {
            rankBuffer = Math.max(0, rankBuffer - 2.25f);
        }
    }

    private void checkLimitedRotation(float dy, float dp) {
        PlayerProfile profile = check.getProfile();

        if (dy < 0.25 || dp < 0.25 || dy > 20.0 || dp > 20.0) {
            return;
        }
        long expandedPitch = (long) (EXPANDER * dp);
        long expandedLastPitch = (long) (EXPANDER * lastDp);
        long gcd = getGcd(expandedPitch, expandedLastPitch);

        if (gcd < 131072L && lastDp > 0.25) {
            patternBuffer += 0.5f;
            if (patternBuffer > 10.0f) {
                profile.punish("Aim", "ProjectAIM",
                        String.format("GCD gcd=%d", gcd), 2.0f);
                patternBuffer = 8.0f;
            }
        }
    }


    private void decayBuffers() {
        rankBuffer = Math.max(0, rankBuffer - 0.05f);
        linearBuffer = Math.max(0, linearBuffer - 0.08f);
        patternBuffer = Math.max(0, patternBuffer - 0.1f);
    }

    private List<Float> getJiffDelta(List<Float> data, int depth) {
        List<Float> result = new ArrayList<>(data);
        for (int i = 0; i < depth; i++) {
            List<Float> next = new ArrayList<>();
            float old = Float.MIN_VALUE;
            for (float n : result) {
                if (old == Float.MIN_VALUE) {
                    old = n;
                    continue;
                }
                next.add(Math.abs(Math.abs(n) - Math.abs(old)));
                old = n;
            }
            result = next;
        }
        return result;
    }


    private int getDistinct(List<Float> data) {
        Set<Float> set = new HashSet<>(data);
        return set.size();
    }


    private double getIQR(List<Float> data) {
        if (data.size() < 4) return 0;
        List<Float> sorted = new ArrayList<>(data);
        sorted.sort(Float::compareTo);
        int n = sorted.size();
        double q1 = sorted.get(n / 4);
        double q3 = sorted.get(3 * n / 4);
        return q3 - q1;
    }


    private List<Double> getZScoreOutliers(List<Double> data, double threshold) {
        if (data.size() < 3) return new ArrayList<>();
        double mean = Statistics.getAverage(data);
        double stdDev = Statistics.getStandardDeviation(data);
        if (stdDev < 1e-6) return new ArrayList<>();
        List<Double> outliers = new ArrayList<>();
        for (double value : data) {
            double zScore = (value - mean) / stdDev;
            if (Math.abs(zScore) > threshold) {
                outliers.add(value);
            }
        }

        return outliers;
    }

    private long getGcd(long a, long b) {
        return (b <= 16384L) ? a : getGcd(b, a % b);
    }
}
