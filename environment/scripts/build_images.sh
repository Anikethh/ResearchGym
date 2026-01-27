#!/bin/bash
# Build ResearchGym Docker Images with GPU Detection

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_BASE=true
BUILD_AGENTS=true
VERBOSE=false
PUSH=false
TAG_PREFIX=""

# Help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build ResearchGym Docker images with GPU-agnostic capabilities.

OPTIONS:
    -h, --help              Show this help message
    --base-only             Build only the base image
    --agents-only           Build only the agent images (requires base)
    --tag-prefix PREFIX     Add a prefix to image tags (e.g., "dev-")
    --push                  Push images to registry after building
    --verbose               Verbose build output
    
EXAMPLES:
    $(basename "$0")                        # Build all images
    $(basename "$0") --base-only            # Build only base image
    $(basename "$0") --tag-prefix dev-      # Tag images with dev- prefix
    $(basename "$0") --push                 # Build and push to registry
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --base-only)
            BUILD_AGENTS=false
            shift
            ;;
        --agents-only)
            BUILD_BASE=false
            shift
            ;;
        --tag-prefix)
            TAG_PREFIX="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            show_help
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINERS_DIR="$REPO_ROOT/environment/containers"

echo -e "${BLUE}ResearchGym Docker Image Builder${NC}"
echo -e "${BLUE}Repository: $REPO_ROOT${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "$CONTAINERS_DIR/Dockerfile.base" ]]; then
    echo -e "${RED}Error: Could not find Dockerfile.base at $CONTAINERS_DIR${NC}" >&2
    echo "Make sure you're running this script from the ResearchGym repository."
    exit 1
fi

# Change to repo root for Docker context
cd "$REPO_ROOT"

# Build base image
if [[ $BUILD_BASE == true ]]; then
    echo -e "${GREEN}Building base image...${NC}"
    BASE_TAG="${TAG_PREFIX}researchgym-base:latest"
    
    BUILD_ARGS=()
    if [[ $VERBOSE == true ]]; then
        BUILD_ARGS+=(--progress=plain)
    fi
    
    docker build \
        "${BUILD_ARGS[@]}" \
        -f environment/containers/Dockerfile.base \
        -t "$BASE_TAG" \
        .
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Base image built: $BASE_TAG${NC}"
        if [[ $PUSH == true ]]; then
            echo -e "${YELLOW}Pushing base image...${NC}"
            docker push "$BASE_TAG"
        fi
    else
        echo -e "${RED}✗ Failed to build base image${NC}" >&2
        exit 1
    fi
    echo ""
fi

# Build agent images
if [[ $BUILD_AGENTS == true ]]; then
    # Update base image reference in agent Dockerfiles if using custom tag
    if [[ -n "$TAG_PREFIX" ]]; then
        echo -e "${YELLOW}Updating Dockerfile references for tag prefix...${NC}"
        for dockerfile in "$CONTAINERS_DIR"/Dockerfile.*-agent "$CONTAINERS_DIR"/Dockerfile.ai-scientist "$CONTAINERS_DIR"/Dockerfile.ml-master "$CONTAINERS_DIR"/Dockerfile.basic-agent-rl; do
            if [[ -f "$dockerfile" ]]; then
                sed -i.bak "s/FROM researchgym-base:latest/FROM ${TAG_PREFIX}researchgym-base:latest/" "$dockerfile"
            fi
        done
    fi
    
    # Agent images to build
    AGENTS=(
        "basic-agent:environment/containers/Dockerfile.basic-agent"
        "basic-agent-rl:environment/containers/Dockerfile.basic-agent-rl"
        "ai-scientist:environment/containers/Dockerfile.ai-scientist" 
        "ml-master:environment/containers/Dockerfile.ml-master"
    )
    
    for agent_info in "${AGENTS[@]}"; do
        IFS=':' read -r agent_name dockerfile <<< "$agent_info"
        
        echo -e "${GREEN}Building $agent_name image...${NC}"
        AGENT_TAG="${TAG_PREFIX}researchgym-$agent_name:latest"
        
        BUILD_ARGS=()
        if [[ $VERBOSE == true ]]; then
            BUILD_ARGS+=(--progress=plain)
        fi
        
        docker build \
            "${BUILD_ARGS[@]}" \
            -f "$dockerfile" \
            -t "$AGENT_TAG" \
            .
        
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ $agent_name image built: $AGENT_TAG${NC}"
            if [[ $PUSH == true ]]; then
                echo -e "${YELLOW}Pushing $agent_name image...${NC}"
                docker push "$AGENT_TAG"
            fi
        else
            echo -e "${RED}✗ Failed to build $agent_name image${NC}" >&2
            # Restore original Dockerfiles if they were modified
            if [[ -n "$TAG_PREFIX" ]]; then
                for dockerfile in "$CONTAINERS_DIR"/Dockerfile.*-agent "$CONTAINERS_DIR"/Dockerfile.ai-scientist "$CONTAINERS_DIR"/Dockerfile.ml-master "$CONTAINERS_DIR"/Dockerfile.basic-agent-rl; do
                    if [[ -f "$dockerfile.bak" ]]; then
                        mv "$dockerfile.bak" "$dockerfile"
                    fi
                done
            fi
            exit 1
        fi
        echo ""
    done
    
    # Restore original Dockerfiles if they were modified
    if [[ -n "$TAG_PREFIX" ]]; then
        echo -e "${YELLOW}Restoring original Dockerfiles...${NC}"
        for dockerfile in "$CONTAINERS_DIR"/Dockerfile.*-agent "$CONTAINERS_DIR"/Dockerfile.ai-scientist "$CONTAINERS_DIR"/Dockerfile.ml-master "$CONTAINERS_DIR"/Dockerfile.basic-agent-rl; do
            if [[ -f "$dockerfile.bak" ]]; then
                mv "$dockerfile.bak" "$dockerfile"
            fi
        done
    fi
fi

echo -e "${GREEN}Build complete!${NC}"

# Show built images
echo -e "${BLUE}Built images:${NC}"
if [[ $BUILD_BASE == true ]]; then
    echo "  - ${TAG_PREFIX}researchgym-base:latest"
fi
if [[ $BUILD_AGENTS == true ]]; then
    echo "  - ${TAG_PREFIX}researchgym-basic-agent:latest"
    echo "  - ${TAG_PREFIX}researchgym-basic-agent-rl:latest"
    echo "  - ${TAG_PREFIX}researchgym-ai-scientist:latest"
    echo "  - ${TAG_PREFIX}researchgym-ml-master:latest"
fi

echo ""
echo -e "${BLUE}Usage examples:${NC}"
echo "  # Run with GPU support (if available)"
echo "  docker run --gpus all -it ${TAG_PREFIX}researchgym-base:latest"
echo ""
echo "  # Run CPU-only"
echo "  docker run -it ${TAG_PREFIX}researchgym-base:latest"
echo ""
echo -e "${GREEN}The images will automatically detect and install appropriate GPU/CPU packages!${NC}"
