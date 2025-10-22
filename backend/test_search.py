"""
Test script for the search pipeline
"""
import asyncio
import json
from loguru import logger

from app.services.search_service import search_service
from app.utils.text_normalizer import text_normalizer


async def test_text_search():
    """Test text search functionality"""
    logger.info("Testing text search...")
    
    # Test queries
    test_queries = [
        "người đi bộ trên đường",
        "xe máy trong thành phố",
        "cửa hàng mở cửa",
        "công viên có cây xanh"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        try:
            results = await search_service.search(
                query=query,
                query_type="text",
                top_k=5,
                use_rag=False
            )
            
            logger.info(f"Found {results['total_results']} results")
            logger.info(f"Search time: {results.get('search_time', 0):.2f}s")
            
            # Show first result if available
            if results['scenes']:
                first_result = results['scenes'][0]
                logger.info(f"Top result: {first_result.get('scene_id', 'N/A')} (score: {first_result.get('score', 0):.3f})")
            
        except Exception as e:
            logger.error(f"Text search failed for '{query}': {str(e)}")


async def test_text_normalization():
    """Test Vietnamese text normalization"""
    logger.info("Testing text normalization...")
    
    test_texts = [
        "Người đi bộ trên đường phố Hà Nội",
        "Xe máy chạy trong thành phố Hồ Chí Minh",
        "Cửa hàng ABC đang mở cửa",
        "Công viên có nhiều cây xanh"
    ]
    
    for text in test_texts:
        normalized = text_normalizer.normalize_text(text)
        keywords = text_normalizer.extract_keywords(text)
        
        logger.info(f"Original: {text}")
        logger.info(f"Normalized: {normalized}")
        logger.info(f"Keywords: {keywords}")
        logger.info("---")


async def test_search_service_initialization():
    """Test search service initialization"""
    logger.info("Testing search service initialization...")
    
    try:
        # Test embedding model
        text_embedding = await search_service.embedding_model.encode_text("test")
        logger.info(f"Text embedding shape: {text_embedding.shape}")
        
        # Test vector database
        stats = search_service.vector_db.get_statistics()
        logger.info(f"Vector DB stats: {stats}")
        
        # Test metadata database
        metadata_stats = await search_service.metadata_db.get_statistics()
        logger.info(f"Metadata DB stats: {metadata_stats}")
        
    except Exception as e:
        logger.error(f"Initialization test failed: {str(e)}")


async def main():
    """Main test function"""
    logger.info("Starting search pipeline tests...")
    
    # Test initialization
    await test_search_service_initialization()
    
    # Test text normalization
    await test_text_normalization()
    
    # Test text search
    await test_text_search()
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
