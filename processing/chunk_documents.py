import json
import os
import re
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Optimized parameters for French administrative texts
CHUNK_SIZE = 1000
OVERLAP = 200  # Increased from 150 to 200 for better continuity

def extract_chapter(url: str) -> str:
    """Extracts the chapter number from the URL."""
    match = re.search(r'chapitre-(\d+)', url)
    return f"Chapter {match.group(1)}" if match else "Uncategorized"

def clean_chunk_text(text: str) -> str:
    """Cleans the text of a chunk."""
    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove multiple line breaks (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def validate_chunk(chunk_text: str, min_length: int = 50) -> bool:
    """Checks whether a chunk is valid (not too short, not empty)."""
    return len(chunk_text.strip()) >= min_length

def main():
    print("\n" + "="*80)
    print("🔄 UQAC DOCUMENT CHUNKING")
    print("="*80 + "\n")
    
    # 1. Load the raw JSON
    input_path = "../data/raw_texts.json"
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        print(f"   Expected location: {os.path.abspath(input_path)}")
        return

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON read error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return

    print(f"✅ {len(raw_documents)} documents loaded")
    
    # 2. Convert to LangChain "Document" objects with ALL metadata
    docs = []
    for doc in raw_documents:
        # IMPROVEMENT 1: Keep ALL metadata, not just the URL
        metadata = {
            "source": doc["metadata"]["source"],
            "title": doc["metadata"].get("title", "Untitled"),
            "doc_type": doc["metadata"].get("doc_type", "Document"),
            "hierarchy": doc["metadata"].get("hierarchy", ""),
            "last_updated": doc["metadata"].get("last_updated", ""),
            "chapitre": extract_chapter(doc["metadata"]["source"]),  # NEW
        }
        
        # Add page_count for PDFs
        if "page_count" in doc["metadata"]:
            metadata["page_count"] = doc["metadata"]["page_count"]
        
        docs.append(Document(
            page_content=doc["content"],
            metadata=metadata
        ))

    print(f"📄 Documents converted with enriched metadata")

    # 3. Smart splitting (Recursive)
    # IMPROVEMENT 2: Separators optimized for French administrative writing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        length_function=len,
        separators=[
            "\n\n\n",  # Major section breaks
            "\n\n",    # Paragraphs
            "\n",      # Lines
            ". ",      # Sentences (with space after the dot)
            "! ",      # Exclamations
            "? ",      # Questions
            ";",       # Semicolons
            ",",       # Commas (last resort)
            " ",       # Spaces
            ""         # Individual characters (absolute last resort)
        ],
        is_separator_regex=False
    )

    print(f"🔪 Splitting in progress (size: {CHUNK_SIZE}, overlap: {OVERLAP})...")
    all_chunks = text_splitter.split_documents(docs)

    # IMPROVEMENT 3: Enrich chunk metadata
    enriched_chunks = []
    chunk_stats = defaultdict(int)
    chunks_by_doc = defaultdict(int)
    
    for i, chunk in enumerate(all_chunks):
        # Text cleaning
        cleaned_text = clean_chunk_text(chunk.page_content)
        
        # Validation
        if not validate_chunk(cleaned_text):
            print(f"⚠️  Chunk {i} too short ({len(cleaned_text)} chars), skipped")
            continue
        
        # Metadata enrichment
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(cleaned_text)
        
        # Statistics
        chunk_stats[chunk.metadata["doc_type"]] += 1
        chunks_by_doc[chunk.metadata["source"]] += 1
        
        enriched_chunks.append({
            "id": f"chunk_{i}",  # Unique ID for ChromaDB
            "text": cleaned_text,
            "metadata": chunk.metadata
        })

    # 4. Detailed statistics
    print("\n" + "="*80)
    print("📊 CHUNKING STATISTICS")
    print("="*80 + "\n")
    
    print(f"📦 Total chunks: {len(enriched_chunks)}")
    print(f"📄 Source documents: {len(raw_documents)}")
    print(f"📈 Average: {len(enriched_chunks) / len(raw_documents):.1f} chunks/document")
    
    # Length distribution
    lengths = [chunk["metadata"]["chunk_length"] for chunk in enriched_chunks]
    print(f"\n📏 Length distribution:")
    print(f"   - Min:     {min(lengths):,} characters")
    print(f"   - Max:     {max(lengths):,} characters")
    print(f"   - Average: {sum(lengths) // len(lengths):,} characters")
    print(f"   - Median:  {sorted(lengths)[len(lengths)//2]:,} characters")
    
    # Distribution by document type
    print(f"\n📋 Chunks by document type:")
    for doc_type, count in sorted(chunk_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(enriched_chunks)) * 100
        print(f"   - {doc_type:20s}: {count:4d} ({percentage:5.1f}%)")
    
    # Top documents with the most chunks
    print(f"\n🔝 Top 5 documents with the most chunks:")
    top_docs = sorted(chunks_by_doc.items(), key=lambda x: x[1], reverse=True)[:5]
    for url, count in top_docs:
        # Shorten the URL for display
        short_url = url.split('/')[-2] if len(url.split('/')) > 2 else url
        print(f"   - {count:3d} chunks: {short_url[:60]}...")
    
    # 5. Saving
    os.makedirs("../data", exist_ok=True)
    
    # Save full chunks (for ChromaDB)
    output_path = "../data/chunks.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Chunks saved: {output_path}")
    except Exception as e:
        print(f"\n❌ Save error: {e}")
        return
    
    # IMPROVEMENT 4: Save metadata separately (for inspection)
    metadata_path = "../data/chunks_metadata.json"
    metadata_summary = {
        "total_chunks": len(enriched_chunks),
        "total_documents": len(raw_documents),
        "avg_chunks_per_doc": len(enriched_chunks) / len(raw_documents),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": OVERLAP,
        "stats_by_type": dict(chunk_stats),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) // len(lengths),
            "median": sorted(lengths)[len(lengths)//2]
        }
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")
    
    # 6. Sample for verification
    print("\n" + "="*80)
    print("📝 SAMPLE (first 3 chunks)")
    print("="*80 + "\n")
    
    for i, chunk in enumerate(enriched_chunks[:3], 1):
        print(f"{i}. ID: {chunk['id']}")
        print(f"   Source: {chunk['metadata']['source'][-60:]}")
        print(f"   Type: {chunk['metadata']['doc_type']}")
        print(f"   Chapter: {chunk['metadata']['chapitre']}")
        print(f"   Length: {chunk['metadata']['chunk_length']} characters")
        preview = chunk['text'][:150].replace('\n', ' ')
        print(f"   Preview: {preview}...")
        print()
    
    print("="*80)
    print("✅ CHUNKING COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    print("💡 Next steps:")
    print("   1. Check chunks.json")
    print("   2. Generate embeddings")
    print("   3. Load into ChromaDB")

if __name__ == "__main__":
    main()
