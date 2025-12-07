
import org.alex.ac.data.PlayerProfile;
import org.bukkit.Location;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;
import org.bukkit.util.Vector;

public final class Direction {

    private static final double DIRECTION_PRECISION = 2.6;
    private static final float[] SIN_TABLE = new float[65536];
    private static final float SIN_SCALE = 10430.378F;

    static {
        for (int i = 0; i < SIN_TABLE.length; ++i) {
            SIN_TABLE[i] = (float) StrictMath.sin((double) i * Math.PI * 2.0 / 65536.0);
        }
    }
    private final PlayerProfile profile;
    private float buffer = 0.0f;
    public Direction(PlayerProfile profile) {
        this.profile = profile;
    }
    public boolean check(Player player, Entity damaged) {
        if (damaged == null || !damaged.isValid()) {
            return false;
        }
        Location playerLoc = player.getLocation();
        Location damagedLoc = damaged.getLocation();
        double width = 0.6;
        double height = 1.8;

        if (damaged instanceof LivingEntity) {
            LivingEntity living = (LivingEntity) damaged;
            height = living.getEyeHeight() * 1.1;
        }

        Vector direction = getLookingDirection(playerLoc.getYaw(), playerLoc.getPitch());
        double off = directionCheck(
                playerLoc,
                player.getEyeHeight(),
                direction,
                damagedLoc.getX(),
                damagedLoc.getY() + height / 2.0,
                damagedLoc.getZ(),
                width,
                height,
                DIRECTION_PRECISION
        );

        boolean cancel = false;

        if (off > 0.1) {
            Vector blockEyes = new Vector(
                    damagedLoc.getX() - playerLoc.getX(),
                    damagedLoc.getY() + height / 2.0 - playerLoc.getY() - player.getEyeHeight(),
                    damagedLoc.getZ() - playerLoc.getZ()
            );

            double distance = blockEyes.crossProduct(direction).length() / direction.length();

            buffer += (float) (distance * 0.8);

            if (buffer > 8.0f) {
                profile.punish("Fight", "Direction",
                        String.format("off=%.2f dist=%.2f", off, distance), 2.0f);
                cancel = true;
                buffer = 6.0f;
            }
        } else {
            buffer = Math.max(0, buffer - 1.5f);
        }

        return cancel;
    }

    private double directionCheck(Location loc, double eyeHeight, Vector direction,
                                   double targetX, double targetY, double targetZ,
                                   double width, double height, double precision) {
        double eyeX = loc.getX();
        double eyeY = loc.getY() + eyeHeight;
        double eyeZ = loc.getZ();

        double dirX = direction.getX();
        double dirY = direction.getY();
        double dirZ = direction.getZ();

        double dirLength = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
        if (dirLength > 0) {
            dirX /= dirLength;
            dirY /= dirLength;
            dirZ /= dirLength;
        }
        double halfWidth = width / 2.0;
        double halfHeight = height / 2.0;
        double minX = targetX - halfWidth;
        double maxX = targetX + halfWidth;
        double minY = targetY - halfHeight;
        double maxY = targetY + halfHeight;
        double minZ = targetZ - halfWidth;
        double maxZ = targetZ + halfWidth;
        double closestX = Math.max(minX, Math.min(eyeX, maxX));
        double closestY = Math.max(minY, Math.min(eyeY, maxY));
        double closestZ = Math.max(minZ, Math.min(eyeZ, maxZ));
        double toTargetX = closestX - eyeX;
        double toTargetY = closestY - eyeY;
        double toTargetZ = closestZ - eyeZ;

        double toTargetLength = Math.sqrt(toTargetX * toTargetX + toTargetY * toTargetY + toTargetZ * toTargetZ);

        if (toTargetLength < 0.001) {
            return 0.0;
        }

        toTargetX /= toTargetLength;
        toTargetY /= toTargetLength;
        toTargetZ /= toTargetLength;

        double dot = dirX * toTargetX + dirY * toTargetY + dirZ * toTargetZ;
        dot = Math.max(-1.0, Math.min(1.0, dot));
        double angle = Math.acos(dot);
        double angleDegrees = Math.toDegrees(angle);
        double allowedAngle = Math.toDegrees(Math.atan2(Math.max(halfWidth, halfHeight), toTargetLength)) + precision;
        if (angleDegrees <= allowedAngle) {
            return 0.0;
        }
        return angleDegrees - allowedAngle;
    }

    private Vector getLookingDirection(float yaw, float pitch) {
        float f = pitch * 0.017453292F;
        float f1 = -yaw * 0.017453292F;
        float f2 = cos(f1);
        float f3 = sin(f1);
        float f4 = cos(f);
        float f5 = sin(f);
        return new Vector(f3 * f4, -f5, f2 * f4);
    }

    private static float sin(float value) {
        return SIN_TABLE[(int) (value * SIN_SCALE) & 0xFFFF];
    }

    private static float cos(float value) {
        return SIN_TABLE[(int) (value * SIN_SCALE + 16384.0F) & 0xFFFF];
    }
}
