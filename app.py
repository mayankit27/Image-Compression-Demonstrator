
import heapq
import io
import struct
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from PIL import Image


# ==================================================
# 1. HUFFMAN TREE NODE
# ==================================================

@dataclass
class Node:
    frequency: int
    pixel_value: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


# ==================================================
# 2. BUILD HUFFMAN TREE
# ==================================================

def build_huffman_tree(frequencies):
    """
    Build a Huffman tree using a min-priority queue.

    Time complexity: O(K log K)
    K = number of distinct pixel values
    """

    heap = []
    counter = 0

    # Insert each pixel value into the priority queue.
    for pixel, frequency in sorted(frequencies.items()):
        node = Node(frequency, pixel)

        # Counter avoids comparing Node objects when
        # two frequencies are equal.
        heapq.heappush(heap, (frequency, counter, node))
        counter += 1

    if not heap:
        return None

    # Handle an image containing only one pixel value.
    if len(heap) == 1:
        return heap[0][2]

    # Repeatedly combine the two least-frequent nodes.
    while len(heap) > 1:
        freq1, _, left = heapq.heappop(heap)
        freq2, _, right = heapq.heappop(heap)

        parent = Node(
            frequency=freq1 + freq2,
            left=left,
            right=right
        )

        heapq.heappush(
            heap,
            (parent.frequency, counter, parent)
        )
        counter += 1

    return heap[0][2]


# ==================================================
# 3. GENERATE HUFFMAN CODES
# ==================================================

def generate_codes(root):
    """
    Generate a binary code for each pixel value.
    Left branch = 0
    Right branch = 1
    """

    codes = {}

    def traverse(node, code):
        if node is None:
            return

        # Leaf node represents a pixel value.
        if node.pixel_value is not None:
            codes[node.pixel_value] = code or "0"
            return

        traverse(node.left, code + "0")
        traverse(node.right, code + "1")

    traverse(root, "")
    return codes


# ==================================================
# 4. PACK BITS INTO BYTES
# ==================================================

def pack_bits(bit_string):
    """
    Convert a binary string into bytes.
    Padding is added when the length is not
    divisible by eight.
    """

    padding = (-len(bit_string)) % 8

    padded_bits = bit_string + ("0" * padding)

    packed_data = bytes(
        int(padded_bits[i:i + 8], 2)
        for i in range(0, len(padded_bits), 8)
    )

    return packed_data, padding


def unpack_bits(packed_data, padding):
    """Convert packed bytes back into the original bits."""

    bits = "".join(
        format(byte, "08b")
        for byte in packed_data
    )

    if padding:
        bits = bits[:-padding]

    return bits


# ==================================================
# 5. ENCODE PIXELS
# ==================================================

def encode_pixels(pixels, codes):
    """
    Replace each pixel with its Huffman code.

    Time complexity: O(N + L)
    N = number of pixels
    L = number of encoded bits
    """

    bit_string = "".join(
        codes[pixel] for pixel in pixels
    )

    packed_data, padding = pack_bits(bit_string)

    return packed_data, len(bit_string), padding


# ==================================================
# 6. DECODE PIXELS
# ==================================================

def decode_pixels(
    packed_data,
    padding,
    root,
    expected_pixel_count
):
    """Reconstruct the original grayscale pixel values."""

    if expected_pixel_count == 0:
        return b""

    # Special case: only one distinct pixel value.
    if root.pixel_value is not None:
        return bytes(
            [root.pixel_value] * expected_pixel_count
        )

    bits = unpack_bits(packed_data, padding)

    decoded_pixels = bytearray()
    current = root

    for bit in bits:
        if bit == "0":
            current = current.left
        else:
            current = current.right

        # Reached a leaf, so one pixel is decoded.
        if current.pixel_value is not None:
            decoded_pixels.append(current.pixel_value)
            current = root

            if len(decoded_pixels) == expected_pixel_count:
                break

    if len(decoded_pixels) != expected_pixel_count:
        raise ValueError("Decoded pixel count is incorrect.")

    return bytes(decoded_pixels)


# ==================================================
# 7. SERIALIZE HUFFMAN TREE FOR SIZE ACCOUNTING
# ==================================================

def serialize_tree(root):
    """
    Preorder tree representation:
    1 = internal node
    0 + 8 bits = leaf pixel value
    """

    bits = []

    def traverse(node):
        if node.pixel_value is not None:
            bits.append("0")
            bits.append(format(node.pixel_value, "08b"))
        else:
            bits.append("1")
            traverse(node.left)
            traverse(node.right)

    traverse(root)
    return "".join(bits)


# ==================================================
# 8. COMPLETE COMPRESSION ANALYSIS
# ==================================================

def compress_image(image):
    # Convert input to 8-bit grayscale.
    grayscale = image.convert("L")

    width, height = grayscale.size
    original_pixels = grayscale.tobytes()

    # Start timing the algorithm.
    start_time = time.perf_counter()

    # Step 1: Count pixel frequencies.
    frequencies = Counter(original_pixels)

    # Step 2: Build the Huffman tree.
    root = build_huffman_tree(frequencies)

    # Step 3: Generate binary codes.
    codes = generate_codes(root)

    # Step 4: Encode pixels.
    encoded_data, encoded_bits, padding = encode_pixels(
        original_pixels, codes
    )

    # Step 5: Decode the compressed data.
    decoded_pixels = decode_pixels(
        encoded_data,
        padding,
        root,
        len(original_pixels)
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Reconstruct the grayscale image.
    reconstructed = Image.frombytes(
        "L",
        (width, height),
        decoded_pixels
    )

    # Calculate the serialized tree overhead.
    tree_bits = serialize_tree(root)
    packed_tree, _ = pack_bits(tree_bits)

    # Simple container format:
    # 4 bytes magic + 4 width + 4 height
    # + 1 padding + 4 tree bit length
    header_size = 17

    complete_size = (
        header_size
        + len(packed_tree)
        + len(encoded_data)
    )

    raw_size = len(original_pixels)
    payload_size = len(encoded_data)

    # Compare against raw grayscale pixel bytes.
    saving_percent = (
        (raw_size - complete_size) / raw_size * 100
        if raw_size else 0
    )

    payload_saving_percent = (
        (raw_size - payload_size) / raw_size * 100
        if raw_size else 0
    )

    return {
        "grayscale": grayscale,
        "reconstructed": reconstructed,
        "frequencies": frequencies,
        "codes": codes,
        "raw_size": raw_size,
        "payload_size": payload_size,
        "complete_size": complete_size,
        "encoded_bits": encoded_bits,
        "tree_bits": len(tree_bits),
        "elapsed_ms": elapsed_ms,
        "saving_percent": saving_percent,
        "payload_saving_percent": payload_saving_percent,
        "lossless": original_pixels == decoded_pixels,
        "width": width,
        "height": height,
    }


# ==================================================
# 9. STREAMLIT USER INTERFACE
# ==================================================

st.set_page_config(
    page_title="Image Compression Demonstrator",
    page_icon="🗜️",
    layout="wide"
)

st.title("Image Compression Demonstrator")
st.caption(
    "Algorithm Analysis and Design | Greedy Algorithm | Huffman Coding"
)

st.write(
    "Upload an image, compress its grayscale pixels using "
    "Huffman Coding, and verify lossless reconstruction."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is None:
    st.info("Upload an image to start compression.")

    st.markdown("""
    ### What you will learn

    - Greedy algorithm
    - Min-priority queue using `heapq`
    - Huffman tree construction
    - Lossless encoding and decoding
    - Compression ratio and execution time
    - Algorithm time complexity
    """)

else:
    try:
        image = Image.open(uploaded_file)

        result = compress_image(image)

        # ------------------------------------------
        # Original and grayscale images
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Grayscale Image")
            st.image(
                result["grayscale"],
                use_container_width=True
            )

        st.divider()

        # ------------------------------------------
        # Compression metrics
        # ------------------------------------------

        st.subheader("Compression Results")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Original Pixel Data",
            f'{result["raw_size"]:,} bytes'
        )

        m2.metric(
            "Encoded Payload",
            f'{result["payload_size"]:,} bytes'
        )

        m3.metric(
            "Complete Huffman Data",
            f'{result["complete_size"]:,} bytes'
        )

        m4.metric(
            "Space Saved",
            f'{result["saving_percent"]:.2f}%'
        )

        st.caption(
            "Sizes are compared against raw grayscale pixels. "
            "Complete Huffman data includes the encoded payload, "
            "serialized tree, and a 17-byte header. The header and "
            "tree are accounted for in the size estimate; this is "
            "not a general-purpose image file format."
        )

        if result["saving_percent"] > 0:
            st.success(
                "The complete Huffman representation is smaller "
                "than the raw grayscale pixel data."
            )
        elif result["saving_percent"] < 0:
            st.warning(
                "The complete Huffman representation is larger "
                "than the raw pixels. Tree and header overhead can "
                "make some images larger."
            )
        else:
            st.info(
                "The complete representation is the same size "
                "as the raw grayscale pixel data."
            )

        # ------------------------------------------
        # Additional measurements
        # ------------------------------------------

        st.subheader("Algorithm Measurements")

        a, b, c = st.columns(3)

        a.metric(
            "Encoded Bitstream",
            f'{result["encoded_bits"]:,} bits'
        )

        b.metric(
            "Distinct Pixel Values",
            str(len(result["frequencies"]))
        )

        c.metric(
            "Execution Time",
            f'{result["elapsed_ms"]:.3f} ms'
        )

        # ------------------------------------------
        # Reconstructed image
        # ------------------------------------------

        st.divider()
        st.subheader("Lossless Reconstruction")

        if result["lossless"]:
            st.success(
                "Verification successful: every reconstructed "
                "grayscale pixel matches the original grayscale pixel."
            )
        else:
            st.error("Reconstruction verification failed.")

        left, right = st.columns(2)

        with left:
            st.write("Original grayscale image")
            st.image(
                result["grayscale"],
                use_container_width=True
            )

        with right:
            st.write("Reconstructed image")
            st.image(
                result["reconstructed"],
                use_container_width=True
            )

        # Download reconstructed image.
        buffer = io.BytesIO()

        result["reconstructed"].save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="Download Reconstructed Image",
            data=buffer.getvalue(),
            file_name="reconstructed_grayscale.png",
            mime="image/png"
        )

        # ------------------------------------------
        # Huffman code table
        # ------------------------------------------

        with st.expander("View Pixel Frequencies and Huffman Codes"):
            rows = []

            for pixel, frequency in sorted(
                result["frequencies"].items()
            ):
                rows.append({
                    "Pixel Value": pixel,
                    "Frequency": frequency,
                    "Huffman Code": result["codes"][pixel],
                    "Code Length": len(result["codes"][pixel])
                })

            st.dataframe(
                rows,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Pixel value 0 represents black and 255 represents white. "
                "More frequent values generally receive shorter codes."
            )

        # ------------------------------------------
        # AAD complexity analysis
        # ------------------------------------------

        with st.expander("AAD Algorithm Analysis"):
            st.markdown("""
            **1. Frequency counting — O(N)**

            Each pixel is visited once to count its frequency.

            **2. Huffman tree construction — O(K log K)**

            A min-priority queue repeatedly selects the two nodes
            with the lowest frequencies and combines them.

            **3. Code generation — O(K)**

            The tree is traversed to assign a binary code to each
            distinct pixel value.

            **4. Encoding and decoding — O(N + L)**

            Encoding processes N pixels and produces L bits.
            Decoding traverses the encoded bits to reconstruct N pixels.

            **Notation**

            - N = total number of pixels
            - K = distinct pixel values, at most 256
            - L = total number of encoded bits

            The displayed execution time is a practical measurement
            of frequency counting, tree construction, code generation,
            encoding, and decoding. It may vary by image and computer.
            """)

    except Exception as error:
        st.error(f"Unable to process this image: {error}")
        st.info("Please try another valid image file.")
