#!/bin/bash
# SoftMig installation script
# Sets up directories, copies library, and configures /etc/ld.so.preload

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
LIB_SOURCE="${1:-/global/home/rahimk/SoftMig/build/libsoftmig.so}"
LIB_DEST="/var/lib/shared/libsoftmig.so"
PRELOAD_FILE="/etc/ld.so.preload"

echo -e "${GREEN}SoftMig Installation Script${NC}"
echo "================================"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root${NC}"
   exit 1
fi

# Check if source library exists
if [[ ! -f "$LIB_SOURCE" ]]; then
    echo -e "${RED}Error: Source library not found: $LIB_SOURCE${NC}"
    echo "Usage: $0 [path/to/libsoftmig.so]"
    exit 1
fi

# Check if library is already in /etc/ld.so.preload
LIB_IN_PRELOAD=false
if [[ -f "$PRELOAD_FILE" ]] && grep -q "^${LIB_DEST}$" "$PRELOAD_FILE" 2>/dev/null; then
    LIB_IN_PRELOAD=true
    echo -e "${YELLOW}Warning: Library is already in /etc/ld.so.preload${NC}"
    echo "Temporarily disabling to allow safe installation..."
    
    # Comment out the line temporarily
    sed -i "s|^${LIB_DEST}$|#${LIB_DEST}|" "$PRELOAD_FILE"
    echo "Library temporarily disabled in /etc/ld.so.preload"
fi

# 1. Create directories
echo "Creating directories..."
mkdir -p /var/lib/shared
mkdir -p /var/log/softmig
mkdir -p /var/run/softmig

# 2. Copy library (using install command which is safer)
echo "Installing library..."
install -m 644 -o root -g root "$LIB_SOURCE" "$LIB_DEST"

# 3. Set ownership and permissions
echo "Setting ownership and permissions..."
chown root:root /var/lib/shared /var/lib/shared/libsoftmig.so
chown root:root /var/log/softmig
chown root:root /var/run/softmig

chmod 755 /var/lib/shared          # drwxr-xr-x (readable/executable by all)
chmod 644 /var/lib/shared/libsoftmig.so  # rw-r--r-- (readable by all)

# Log directory: Use group writable (775) with slurm group, or 1777 with sticky bit
# Option 1: Group writable (recommended - more secure)
if getent group slurm >/dev/null 2>&1; then
    chown root:slurm /var/log/softmig
    chmod 775 /var/log/softmig  # drwxrwxr-x (group writable)
    echo "Using slurm group for log directory (775)"
else
    # Fallback: Sticky bit + world writable (1777) - less secure but works
    chmod 1777 /var/log/softmig  # drwxrwxrwt (sticky bit prevents deletion of others' files)
    echo "Using sticky bit for log directory (1777) - consider creating a softmig group"
fi

chmod 755 /var/run/softmig          # drwxr-xr-x (readable by all, writable only by root)

# 4. Add to /etc/ld.so.preload (if not already there)
echo "Configuring /etc/ld.so.preload..."
if [[ "$LIB_IN_PRELOAD" == "false" ]]; then
    if ! grep -q "^${LIB_DEST}$" "$PRELOAD_FILE" 2>/dev/null; then
        echo "$LIB_DEST" >> "$PRELOAD_FILE"
        chmod 644 "$PRELOAD_FILE"  # Set correct permissions (readable by all)
        chown root:root "$PRELOAD_FILE"
        echo -e "${GREEN}Added ${LIB_DEST} to /etc/ld.so.preload${NC}"
    else
        echo -e "${YELLOW}Library already in /etc/ld.so.preload${NC}"
    fi
else
    # Re-enable it (uncomment)
    sed -i "s|^#${LIB_DEST}$|${LIB_DEST}|" "$PRELOAD_FILE"
    chmod 644 "$PRELOAD_FILE"  # Ensure correct permissions
    chown root:root "$PRELOAD_FILE"
    echo -e "${GREEN}Re-enabled library in /etc/ld.so.preload${NC}"
fi

# 5. Verify
echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Verification:"
echo "  Library:"
ls -lh "$LIB_DEST"
echo ""
echo "  Directories:"
ls -ld /var/lib/shared
ls -ld /var/log/softmig
ls -ld /var/run/softmig
echo ""
echo "  /etc/ld.so.preload:"
if grep -q "^${LIB_DEST}$" "$PRELOAD_FILE" 2>/dev/null; then
    echo -e "    ${GREEN}✓ Library is configured${NC}"
    grep "^${LIB_DEST}$" "$PRELOAD_FILE"
else
    echo -e "    ${RED}✗ Library not found in /etc/ld.so.preload${NC}"
fi
echo ""
echo "  Testing user access:"
if sudo -u rahimk ls -l "$LIB_DEST" >/dev/null 2>&1; then
    echo -e "    ${GREEN}✓ Users can access the library${NC}"
else
    echo -e "    ${RED}✗ ERROR: Users cannot access the library!${NC}"
    echo "    Check permissions:"
    ls -ld /var/lib/shared
    ls -l "$LIB_DEST"
fi

