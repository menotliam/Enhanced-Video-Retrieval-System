#!/usr/bin/env python3
"""
Quick CLI to test text-to-frame search using the built FAISS index.

Also exposes a programmatic function `search_frames_programmatic` for API use.
"""

import sys
import asyncio
import json
import unicodedata
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional
import csv


def resolve_project_root() -> None:
    this_file = Path(__file__).resolve()
    backend_dir = this_file.parents[1]  # backend/
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
# Helpers
def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    #print(f"Normalized: {s}")
    return s
    
def _detect_ocr_query(query: str) -> bool:
        """Detect if query is asking for text/OCR content"""
        query_lower = _normalize(query)
        print(f"Query normalized: {query_lower}")
        # Keywords that indicate user wants to find text content
        ocr_keywords = [
            "dòng chữ", "dong chu", "ticker", "subtitle", 
            "caption", "label", "title", "heading", "header", "word",
            "letter", "sentence", "phrase", "written", "printed", "displayed",
            "shown", "appears", "contains", "says", "reads",
            "text banner", "text warning", "text alert", "text notification"
        ]
        
        # Debug: check each keyword
        print("🔍 Checking OCR keywords:")
        matched_keywords = []
        query_words = query_lower.split()
        for keyword in ocr_keywords:
            # Check if keyword is a complete word, not just a substring
            if keyword in query_words:
                matched_keywords.append(keyword)
                print(f"   ✅ MATCH: '{keyword}' found as complete word")
            elif keyword in query_lower:
                # Check if it's a multi-word phrase
                if ' ' in keyword:
                    matched_keywords.append(keyword)
                    print(f"   ✅ MATCH: '{keyword}' found as phrase")
                else:
                    print(f"   ⚠️ PARTIAL MATCH: '{keyword}' found as substring (ignoring)")
            else:
                print(f"   ❌ NO MATCH: '{keyword}' not found")
        
        if matched_keywords:
            print(f"🎯 OCR Query detected! Matched keywords: {matched_keywords}")
        else:
            print("❌ No OCR keywords matched - this is a regular query")
        
        return len(matched_keywords) > 0
def _extract_search_terms(query: str) -> list:
        """Extract meaningful search terms from query, removing OCR keywords"""
        query_lower = _normalize(query)
        
        # Remove OCR keywords to get actual search terms
        ocr_keywords_to_remove = [
            "dòng chữ", "dong chu", "text", "chữ", "chu", "ticker", "subtitle", 
            "caption", "label", "title", "heading", "header", "text", "word",
            "letter", "sentence", "phrase", "written", "printed", "displayed",
            "shown", "appears", "contains", "says", "reads", "written text",
            "printed text", "display text", "text on screen", "text overlay",
            "text banner", "text warning", "text alert", "text notification"
        ]
        
        # Remove OCR keywords
        for keyword in ocr_keywords_to_remove:
            query_lower = query_lower.replace(keyword, "")
        
        # Clean up extra spaces and punctuation
        import re
        query_lower = re.sub(r'\s+', ' ', query_lower).strip()
        query_lower = re.sub(r'[^\w\s]', ' ', query_lower)
        query_lower = re.sub(r'\s+', ' ', query_lower).strip()
        
        # Split into terms
        terms = [term.strip() for term in query_lower.split() if len(term.strip()) >= 2]
        return terms
async def search_frames_programmatic(
    query: str,
    *,
    data_root: str = "data",
    embeddings_root: str = "data/embeddings_v2",
    index_dir: str = "data/faiss_index_v2",
    top_k: int = 10,
    use_gpu: bool = False,
    use_llm: bool = True,
    pre_filter: bool = False,
    pre_filter_limit: int = 0,
) -> Dict[str, Any]:
    """Run the frame search and return structured results.

    Returns a dict: { total_results, results: [ { uid, video_id, frame_idx, pts_time, image_path, score, distance } ] }
    """
    resolve_project_root()

    # Debug: basic inputs
    try:
        print(f"\n🔎 [frames-cli] Query: {query}")
        print(f"📁 [frames-cli] Data root: {data_root}")
        print(f"📦 [frames-cli] Index dir: {index_dir}")
        print(f"🔢 [frames-cli] top_k: {top_k} | use_llm: {use_llm} | pre_filter: {pre_filter}")
    except Exception:
        pass

    # Lazy imports after sys.path resolution
    try:
        from app.db.vector_db import VideoEmbeddingIndex, search_text_clip_only  # noqa: F401
        from embedding_model_2 import init_models
        if use_llm:
            from app.services.llm_service import llm_service  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Import error: {e}")

    # Detect if this is an OCR/text query
    #is_ocr_query = _detect_ocr_query(query)
    is_ocr_query = False
    if is_ocr_query:
        search_terms = _extract_search_terms(query)
    else:
        search_terms = []

    # Initialize models
    print("🤖 [frames-cli] Loading models...")
    models = init_models()
    print("✅ [frames-cli] Models loaded")

    # Load indexes
    index = VideoEmbeddingIndex(data_root=data_root, use_gpu=use_gpu)
    index.load_indexes(index_dir)
    try:
        print(f"🗂️  [frames-cli] Frame index path: {index.frame_index_path}")
    except Exception:
        pass

    # Helper to build enriched terms
    def _build_terms(base_terms: list) -> set:
        # Stopwords/determiners common in Vietnamese noun phrases
        stopwords = {
            "cai", "chiec", "con", "cay", "bo", "nhung", "cac", "mot", "vay", "va", "la",
            "nhom", "bon", "dam", "nhieu", "vai", "thu", "co", "tren", "duoi", "dang",
            "ong","ba"
        }
        terms: set = set()
        
        for term in base_terms:
            t = _normalize(term)
            if not t:
                continue
            # Split into individual words and filter out stopwords
            toks = [tok for tok in t.split() if tok and tok not in stopwords and len(tok) >= 1]
            if len(toks) < 2:
                continue  # Skip if less than 2 tokens (can't form bigrams)
            
            # Add bigrams (consecutive token pairs) only
            for i in range(len(toks) - 1):
                bigram = f"{toks[i]} {toks[i+1]}"
                terms.add(bigram)
        
        print(f"🔧 Terms (bigrams only): {terms}")
        return terms

    # Optionally parse/expand with LLM
    qtext = query
    keywords: List[str] = []
    expanded_terms: List[str] = []
    extracted_filters: List[str] = []
    #etext = qtext
    if use_llm:
        try:
            parsed = await llm_service.parse_and_expand_query(qtext)  # type: ignore
            qtext = parsed.get("normalized_query") or parsed.get("original_query") or qtext
            etext = parsed.get("english_query") or qtext
            print(f"🧠 LLM normalized query: {qtext} | {etext}")
            if isinstance(parsed, dict):
                kws = parsed.get("keywords") or []
                exps = parsed.get("expanded_terms") or parsed.get("expansions") or []
                if isinstance(kws, list):
                    keywords = [str(x) for x in kws]
                if isinstance(exps, list):
                    expanded_terms = [str(x) for x in exps]
                filters = parsed.get("extracted_filters") or []
                if isinstance(filters, list):
                    extracted_filters = [str(x) for x in filters]
        except Exception:
            pass

    should_pre_filter = pre_filter or is_ocr_query

    # Optional pre-filter using metadata JSONs
    allowed_uids = None
    if should_pre_filter:
        try:
            if is_ocr_query and search_terms:
                base = search_terms
            else:
                base = (keywords + expanded_terms + extracted_filters)
                
            terms = _build_terms(base)
            print(f"🔍 Base terms: {terms}")
            if not terms:
                terms = _build_terms([qtext]) or set(token for token in _normalize(qtext).split() if token)

            frame_idx_path = Path(data_root) / "embeddings_v2" / "frame_index.json"
            obj_idx_path = Path(data_root) / "embeddings_v2" / "obj_index.json"
            cand = set()

            if frame_idx_path.exists():
                with open(frame_idx_path, 'r', encoding='utf-8') as f:
                    frame_items = json.load(f)
                for it in frame_items:
                    uid = it.get('uid')
                    if not uid:
                        continue
                    # Always search only in ocr_texts field when pre-filter is enabled
                    ocr_texts_raw = it.get('ocr_texts', [])
                    if isinstance(ocr_texts_raw, list):
                        ocr_texts = ' '.join([_normalize(str(text)) for text in ocr_texts_raw])
                    else:
                        ocr_texts = _normalize(str(ocr_texts_raw))
                    hay = ocr_texts
                    if any(term in hay for term in terms):
                        cand.add(uid)

            # Skip object search since we're only focusing on OCR text filtering

            if pre_filter_limit and pre_filter_limit > 0 and len(cand) > pre_filter_limit:
                cand = set(list(cand)[:pre_filter_limit])
            allowed_uids = cand
        except Exception:
            allowed_uids = None
    #print(f"🔎 Allowed UIDs: {allowed_uids}")
    # Execute search
    if is_ocr_query and allowed_uids and len(allowed_uids) > 0:
        results = []
        frame_idx_path = Path(data_root) / "embeddings_v2" / "frame_index.json"
        if frame_idx_path.exists():
            with open(frame_idx_path, 'r', encoding='utf-8') as f:
                frame_items = json.load(f)
            for uid in list(allowed_uids)[:top_k]:
                for item in frame_items:
                    if item.get('uid') == uid:
                        results.append({
                            'uid': uid,
                            'metadata': item,
                            'score': 1.0,
                            'distance': 0.0,
                            'rank': len(results) + 1,
                            'ocr_match': True
                        })
                        break
    else:
        # Standard semantic search
        results = index.search_by_text_fusion(
            query=qtext,
            top_k=top_k,
            embedding_model=models,
            asr_conf=0.0,
            ocr_conf=0.0 if not is_ocr_query else 0.0,
            scene_score=0.0,
            has_objects=False
        )

    if allowed_uids is not None and results:
        filtered = [r for r in results if r.get('uid') in allowed_uids]
        if filtered:
            results = filtered

    if not results:
        print("❌ [frames-cli] No results found.")
        return {"total_results": 0, "results": []}

    # Format (skip entries with missing metadata)
    formatted: List[Dict[str, Any]] = []
    for r in results[: max(top_k * 2, top_k)]:  # oversample to allow skipping invalid entries
        meta = r.get("metadata") or {}
        video_id = meta.get("video_id")
        frame_idx = meta.get("frame_idx")
        if video_id is None or frame_idx is None:
            continue
        formatted.append({
            "uid": r.get("uid"),
            "video_id": video_id,
            "frame_idx": frame_idx,
            "pts_time": meta.get("pts_time"),
            "image_path": (meta.get("paths", {}) or {}).get("image"),
            "score": r.get("score"),
            "distance": r.get("distance")
        })

    print(f"✅ [frames-cli] Returning {len(formatted)} results")
    return {"total_results": len(formatted), "results": formatted}


def main():
    resolve_project_root()

    parser = argparse.ArgumentParser(description="Search frames by text query using FAISS index")
    parser.add_argument("query", type=str, help="Text query to search")
    parser.add_argument("--data-root", type=str, default="data", help="Root data directory (default: data)")
    parser.add_argument("--embeddings-root", type=str, default="data/embeddings_v2", help="Root embeddings directory (default: data/embeddings_v2)")
    parser.add_argument("--index-dir", type=str, default="data/faiss_index", help="Directory containing FAISS indexes")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return (default: 10)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for FAISS if available")
    parser.add_argument("--use-llm", action="store_true", help="Parse and expand query with LLM before search")
    parser.add_argument("--pre-filter", action="store_true", help="Pre-filter candidates using metadata JSONs")
    parser.add_argument("--pre-filter-limit", type=int, default=0, help="Max candidates to keep from pre-filter (0 = keep all)")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV file (provide filename)")

    args = parser.parse_args()

    # Lazy imports after sys.path resolution
    try:
        from app.db.vector_db import VideoEmbeddingIndex, search_text_clip_only
        from embedding_model_2 import init_models
        if args.use_llm:
            from app.services.llm_service import llm_service
    except Exception as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    print(f"🔎 Query: {args.query}")
    print(f"📁 Data root: {args.data_root}")
    print(f"📦 Index dir: {args.index_dir}")

    # Detect if this is an OCR/text query
    is_ocr_query = _detect_ocr_query(args.query)
    if is_ocr_query:
        print("📝 Detected OCR/Text query - will focus on OCR content")
        search_terms = _extract_search_terms(args.query)
        print(f"🔍 Extracted search terms: {search_terms}")
    else:
        print("🖼️ Regular query - using semantic search")
        search_terms = []

    # Initialize models
    print("🤖 Loading models...")
    models = init_models()
    print("✅ Models loaded")

    # Load indexes
    index = VideoEmbeddingIndex(data_root=args.data_root, use_gpu=args.use_gpu)
    index.load_indexes(args.index_dir)

    

    


    def _build_terms(base_terms: list) -> set:
        # Stopwords/determiners common in Vietnamese noun phrases
        stopwords = {
            "cai", "chiec", "con", "cay", "bo", "nhung", "cac", "mot", "vay", "va", "la",
            "nhom", "bon", "dam", "nhieu", "vai", "thu", "co", "tren", "duoi", "dang",
            "ong","ba"
        }
        terms: set = set()
        
        for term in base_terms:
            t = _normalize(term)
            if not t:
                continue
            toks = [tok for tok in t.split() if tok and tok not in stopwords]
            if not toks:
                continue
            
            # Add unigrams (individual tokens)
            for tok in toks:
                if len(tok) >= 2:  # Skip single characters
                    terms.add(tok)
            
            # Add bigrams (consecutive token pairs)
            for i in range(len(toks) - 1):
                terms.add(f"{toks[i]} {toks[i+1]}")
        
        #print(f"🔧 Terms: {terms}")
        return terms

    # Optionally parse/expand with LLM
    qtext = args.query
    keywords = []
    expanded_terms = []
    extracted_filters = []
    if args.use_llm:
        try:
            # Await async LLM parsing in this CLI context
            parsed = asyncio.run(llm_service.parse_and_expand_query(qtext))
            # Fallbacks
            qtext = parsed.get("normalized_query") or parsed.get("original_query") or qtext
            etext = parsed.get("english_query") or qtext
            print(f"🧠 LLM normalized query: {qtext} | {etext}")
            # Collect filters/keywords/expansions if available
            if isinstance(parsed, dict):
                kws = parsed.get("keywords") or []
                exps = parsed.get("expanded_terms") or parsed.get("expansions") or []
                if isinstance(kws, list):
                    keywords = [str(x) for x in kws]
                if isinstance(exps, list):
                    expanded_terms = [str(x) for x in exps]
                filters = parsed.get("extracted_filters") or []
                if isinstance(filters, list):
                    extracted_filters = [str(x) for x in filters]
        except Exception as e:
            print(f"⚠️ LLM parsing failed, using raw query. Reason: {e}")

    # Auto-enable pre-filter for OCR queries or manual pre-filter
    should_pre_filter = args.pre_filter or is_ocr_query
    
    # Optional pre-filter using metadata JSONs
    allowed_uids = None
    if should_pre_filter:
        try:
            # For OCR queries, use extracted search terms; otherwise use LLM keywords
            if is_ocr_query and search_terms:
                base = search_terms
                print(f"📝 Using extracted OCR search terms: {base}")
            else:
                base = (keywords + expanded_terms + extracted_filters)
                print(f"🧠 Using LLM keywords: {base}")
            
            # Build enriched term set with stopword removal and n-grams
            terms = _build_terms(base)
            print(f"🔧 Terms: {terms}")
            if not terms:
                # fallback to tokens from query text
                terms = _build_terms([qtext]) or set(token for token in _normalize(qtext).split() if token)
            
            frame_idx_path = Path(args.data_root) / "embeddings_v2" / "frame_index.json"
            obj_idx_path = Path(args.data_root) / "embeddings_v2" / "obj_index.json"
            cand = set()
            
            # Scan frame_index.json
            if frame_idx_path.exists():
                with open(frame_idx_path, 'r', encoding='utf-8') as f:
                    frame_items = json.load(f)
                for it in frame_items:
                    uid = it.get('uid')
                    if not uid:
                        continue
                    
                    if is_ocr_query:
                        # For OCR queries, only search in OCR text
                        ocr_texts_raw = it.get('ocr_texts', [])
                        # Handle both list and string formats
                        if isinstance(ocr_texts_raw, list):
                            ocr_texts = ' '.join([_normalize(str(text)) for text in ocr_texts_raw])
                        else:
                            ocr_texts = _normalize(str(ocr_texts_raw))
                        hay = ocr_texts
                        #print(f"🔍 OCR search in: {ocr_texts[:100]}...")
                        
                    else:
                        # For regular queries, search in all text fields
                        asr_text = _normalize(it.get('asr_text', ''))
                        scene_label = _normalize(it.get('scene_label', ''))
                        ocr_texts_raw = it.get('ocr_texts', [])
                        # Handle both list and string formats
                        if isinstance(ocr_texts_raw, list):
                            ocr_texts = ' '.join([_normalize(str(text)) for text in ocr_texts_raw])
                        else:
                            ocr_texts = _normalize(str(ocr_texts_raw))
                        hay = f"{asr_text} {scene_label} {ocr_texts}"
                    
                    if any(term in hay for term in terms):
                        cand.add(uid)
            print(f"🔎 Pre-filter candidates: {cand}")
            # For OCR queries, skip object search since we only want text content
            if not is_ocr_query:
                # Scan obj_index.json (map obj uid -> frame uid)
                if obj_idx_path.exists():
                    with open(obj_idx_path, 'r', encoding='utf-8') as f:
                        obj_items = json.load(f)
                    for it in obj_items:
                        uid = it.get('uid')
                        label_vi = _normalize(it.get('label_vi', ''))
                        if not uid:
                            continue
                        if any(term in label_vi for term in terms):
                            parts = uid.split(':')
                            if len(parts) >= 2:
                                frame_uid = f"{parts[0]}:{parts[1]}"
                                cand.add(frame_uid)
            else:
                print("📝 Skipping object search for OCR query")
            
            #print(f"🔎 Pre-filter candidates: {cand}")
            # Optionally cap if user sets a positive limit
            if args.pre_filter_limit and args.pre_filter_limit > 0 and len(cand) > args.pre_filter_limit:
                cand = set(list(cand)[:args.pre_filter_limit])
            allowed_uids = cand
            print(f"🔎 Pre-filter candidates: {len(allowed_uids)}")
            #print(f"🔎 Pre-filter candidates: {allowed_uids}")
        except Exception as e:
            print(f"⚠️ Pre-filter failed, continuing without it. Reason: {e}")

    # Execute search based on query type
    print(f"Frame index path: {index.frame_index_path}")
    
    if is_ocr_query:
        print("📝 OCR Query detected - using OCR-focused search strategy")
        # For OCR queries, we can skip semantic search if we have good pre-filter results
        if allowed_uids and len(allowed_uids) > 0:
            print(f"✅ Found {len(allowed_uids)} OCR matches via pre-filter, skipping semantic search")
            # Create mock results from pre-filter matches
            results = []
            for uid in list(allowed_uids)[:args.top_k]:
                # Try to get metadata for this UID
                frame_idx_path = Path(args.data_root) / "embeddings_v2" / "frame_index.json"
                if frame_idx_path.exists():
                    with open(frame_idx_path, 'r', encoding='utf-8') as f:
                        frame_items = json.load(f)
                    for item in frame_items:
                        if item.get('uid') == uid:
                            results.append({
                                'uid': uid,
                                'metadata': item,
                                'score': 1.0,  # High score for exact OCR match
                                'distance': 0.0,
                                'rank': len(results) + 1,
                                'ocr_match': True
                            })
                            break
        else:
            print("⚠️ No OCR matches found, falling back to semantic search")
            # Fallback to semantic search with OCR focus
            results = index.search_by_text_fusion(
                query=etext,
                top_k=30,
                embedding_model=models,
                asr_conf=0.0,  # Tắt ASR
                ocr_conf=0.0,  # Bật OCR cho OCR queries
                scene_score=0.0,  # Tắt Scene
                has_objects=False
            )
    else:
        print("🖼️ Regular query - using standard semantic search")
        # Standard semantic search
        results = index.search_by_text_fusion(
            query=etext,
            top_k=args.top_k,
            embedding_model=models,
            asr_conf=0.0,  # Tắt ASR
            ocr_conf=0.0,  # Tắt OCR
            scene_score=0.0,  # Tắt Scene
            has_objects=False
        )
    # Extract frame results (v2 format returns dict with "frames" and "objects" keys)
    #results = search_results.get("frames", []) + search_results.get("objects", [])

    # Apply pre-filter if available
    if allowed_uids is not None and results:
        filtered = [r for r in results if r.get('uid') in allowed_uids]
        if filtered:
            results = filtered
        else:
            print("⚠️ Pre-filter yielded 0 hits in FAISS results. Using unfiltered top-k.")

    if not results:
        print("No results found.")
        return

    print(f"\n=== Top {len(results)} results ===")
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        uid = r.get("uid")
        vid = meta.get("video_id")
        frame_idx = meta.get("frame_idx")
        pts = meta.get("pts_time")
        img = meta.get("paths", {}).get("image")
        score = r.get("score")
        dist = r.get("distance")
        ocr_match = r.get("ocr_match", False)
        
        print(f"{i+1}. uid={uid} | video={vid} frame={frame_idx} time={pts}s")
        print(f"   image={img}")
        print(f"   score={score:.4f} distance={dist:.4f} rank={r.get('rank')}")
        
        if ocr_match:
            print(f"   📝 EXACT OCR MATCH")
        
        # Show OCR text if available and it's an OCR query
        if is_ocr_query and meta.get("ocr_texts"):
            ocr_texts_raw = meta.get("ocr_texts", [])
            if isinstance(ocr_texts_raw, list):
                ocr_text = ' | '.join([str(text) for text in ocr_texts_raw])
            else:
                ocr_text = str(ocr_texts_raw)
            print(f"   📄 OCR Text: {ocr_text[:100]}{'...' if len(ocr_text) > 100 else ''}")
        
        print()

    # Export to CSV if requested
    if args.export_csv:
        try:
            csv_filename = args.export_csv
            if not csv_filename.endswith('.csv'):
                csv_filename += '.csv'
            
            print(f"\n📄 Exporting results to CSV: {csv_filename}")
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
               
                
                # Write data rows
                for r in results:
                    meta = r.get("metadata", {})
                    video_id = meta.get("video_id", "")
                    frame_idx = meta.get("frame_idx", "")
                    writer.writerow([video_id, frame_idx])
            
            print(f"✅ Successfully exported {len(results)} results to {csv_filename}")
            
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")


if __name__ == "__main__":
    main()
