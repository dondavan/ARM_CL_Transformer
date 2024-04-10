/*
 * Copyright (c) 2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef CKW_INCLUDE_CKW_TYPES_TENSORSTORAGETYPE_H
#define CKW_INCLUDE_CKW_TYPES_TENSORSTORAGETYPE_H

#include <cstdint>

namespace ckw
{

/** Compute Kernel Writer tensor storage.
 *  The tensor storage represents the type of tensor memory object.
 */
enum class TensorStorageType : uint32_t
{
    Unknown            = 0x00000000,
    BufferUint8Ptr     = 0x01000000,
    Texture2dReadOnly  = 0x02000001,
    Texture2dWriteOnly = 0x02000010,
};

} // namespace ckw

#endif // CKW_INCLUDE_CKW_TYPES_TENSORSTORAGETYPE_H
