# Third-party licenses and notices

This project derives from, converts, and redistributes material from the
**TLTK** (Thai Language Toolkit) Python package, version 1.10, ports the
**MaltParser** 1.9.2 engine, and uses a third-party font for label rendering.
This file carries the notices that must travel with distributions of this
project. This project is independent and is not affiliated with, endorsed by,
or sponsored by any of the parties below.

## MaltParser — BSD-3-Clause

The `crates/malt-parser` crate is a Rust port of MaltParser 1.9.2's inference
engine (arc-eager transition system, feature model, and the
`MaltLiblinearModel` predictor vendored in MaltParser's source, which itself
implements LIBLINEAR's MCSVM_CS prediction), and `malt.bin.gz` derives from
the `thamalt.mco` model distributed with TLTK. The crate states the port in
its description; no MaltParser source files are redistributed verbatim.
MaltParser declares a BSD-style license, with this notice:

Copyright (c) 2007-2018 Johan Hall, Jens Nilsson and Joakim Nivre

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of MaltParser nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

LIBLINEAR (the algorithm behind the vendored predictor) is
Copyright (c) 2007-2021 The LIBLINEAR Project, BSD-3-Clause; the port touches
only the linear-scan prediction path via MaltParser's Java class above.

## TLTK — BSD-3-Clause

The Rust algorithms in `crates/tltk-core` are a port of TLTK 1.10
(`tltk/nlp.py`, `tltk/corpus.py`), and the model blobs under
`crates/tltk-core/data/` (thdict, sylseg, sylvar, phs, pos, ner, edu, w2v,
sylrule, thaisyl — converted formats) derive from the data files shipped in
that package. The package declares `License: BSD-3-Clause` in its wheel
metadata, with this notice:

Copyright (c) 2018 Dept of Linguistics, Chulalongkorn University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Unidentified GPL-3 file in the TLTK wheel (open question)

The TLTK 1.10 wheel bundles a second license file, `LICENSE-gpl.txt`
(GPL-3.0), without stating which component it covers. If it covers any of the
data files converted into the blobs above (candidates: thaisyl.dict,
sylrule.lts, BEST.dict, thamalt.mco), those specific blobs may need to be
redistributed under GPL terms. Resolve with upstream before commercial
reliance.

## TNC word2vec model

`w2v.bin.gz` is converted from `TNCc5model3.bin`, a word2vec model trained
on the Thai National Corpus (TNC). TNC data may carry its own access/use terms
from Chulalongkorn University independent of the package license above.

## TH Sarabun New font (glyph atlas)

`crates/tltk-core/data/glyph_atlas.png` is a pre-rasterized bitmap atlas
derived from **TH Sarabun New Bold** (the font the TLTK reference charts use).
TH Sarabun New is a free font commonly distributed under the SIL Open Font
License 1.1; the local build used here (`THSarabunNew-Bold-NoPUA.ttf`)
carries no embedded license metadata, so verify the license of your font
source before redistributing the atlas.
