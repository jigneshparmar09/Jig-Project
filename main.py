#main.py
import os

import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from opensearchpy import OpenSearch
from dateutil import parser as date_parser
import dateutil.parser
from dateutil import tz
from dateutil import tz
import dateparser
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Using .env file for configuration")
except ImportError:
    logger.warning("⚠️ python-dotenv not installed. Using manual configuration.")

# Check for optional Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
    logger.info("✅ Ollama package available")
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ Ollama package not installed.")

# Check for requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.error("❌ requests package not available.")

# Config variables
ES_HOST = os.getenv("ES_HOST", "http://sikkapaydashboard.sikkasoft.com:9200")
ES_INDEX = os.getenv("ES_INDEX", "nadapayments")
ES_USERNAME = os.getenv("ES_USERNAME", "User")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))

app = FastAPI(
    title="Kibana AI Assistant",
    description="AI-powered Log Analysis Assistant with Chat Interface",
    version="2.0.0", 
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
try:
    templates = Jinja2Templates(directory="templates")
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Template/static setup failed: {e}")

# Initialize OpenSearch client
def create_opensearch_client():
    """Create OpenSearch connection with proper error handling"""
    
    logger.info("🔍 Creating OpenSearch connection...")
    logger.info(f"Host: {ES_HOST}")
    logger.info(f"Username: {ES_USERNAME}")
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(ES_HOST)
        
        client = OpenSearch(
            hosts=[{
                'host': parsed.hostname,
                'port': parsed.port or 9200,
                'use_ssl': parsed.scheme == 'https'
            }],
            http_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False,
            timeout=30
        )
        
        # Test connection
        info = client.info()
        logger.info("✅ OpenSearch connected successfully!")
        logger.info(f"✅ Cluster: {info['cluster_name']}")
        logger.info(f"✅ Version: {info['version']['number']}")
        
        # Test index
        if ES_INDEX:
            exists = client.indices.exists(index=ES_INDEX)
            logger.info(f"✅ Index '{ES_INDEX}' exists: {exists}")
            
            if exists:
                stats = client.indices.stats(index=ES_INDEX)
                doc_count = stats['indices'][ES_INDEX]['total']['docs']['count']
                logger.info(f"✅ Document count: {doc_count:,}")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ OpenSearch connection failed: {e}")
        return None

es = create_opensearch_client()

# Pydantic models
class QueryRequest(BaseModel):
    prompt: str
    index: Optional[str] = None
    max_results: Optional[int] = None

class PaginatedQueryRequest(BaseModel):
    prompt: str
    index: Optional[str] = None
    max_results: Optional[int] = None
    offset: Optional[int] = 0
    page_size: Optional[int] = 20

def parse_natural_language_query(prompt: str) -> Dict:
    """Parse natural language queries into search parameters"""
    filters = {}
    prompt_lower = prompt.lower()
    # PRIORITY 1: Handle relative dates FIRST
    current_date = datetime.now()
    
    if "today" in prompt_lower:
        today = current_date.strftime("%Y-%m-%d")
        filters["date"] = today
        logger.info(f"📅 TODAY DETECTED: {today}")
    elif "yesterday" in prompt_lower:
        yesterday = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        filters["date"] = yesterday
        logger.info(f"📅 YESTERDAY DETECTED: {yesterday}")
    elif "tomorrow" in prompt_lower:
        tomorrow = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        filters["date"] = tomorrow
        logger.info(f"📅 TOMORROW DETECTED: {tomorrow}")
    elif "last week" in prompt_lower:
        # Calculate last week range (Monday to Sunday)
        days_since_monday = current_date.weekday()
        last_monday = current_date - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        filters["date_range"] = [
            last_monday.strftime("%Y-%m-%d"),
            last_sunday.strftime("%Y-%m-%d")
        ]
        logger.info(f"📅 LAST WEEK DETECTED: {last_monday.strftime('%Y-%m-%d')} to {last_sunday.strftime('%Y-%m-%d')}")
    elif "last month" in prompt_lower:
        # Calculate last month range
        from dateutil.relativedelta import relativedelta
        first_day_this_month = current_date.replace(day=1)
        last_day_last_month = first_day_this_month - timedelta(days=1)
        first_day_last_month = last_day_last_month.replace(day=1)
        filters["date_range"] = [
            first_day_last_month.strftime("%Y-%m-%d"),
            last_day_last_month.strftime("%Y-%m-%d")
        ]
        logger.info(f"📅 LAST MONTH DETECTED: {first_day_last_month.strftime('%Y-%m-%d')} to {last_day_last_month.strftime('%Y-%m-%d')}")
    elif "this month" in prompt_lower:
        # Calculate this month range (1st to today)
        first_day_this_month = current_date.replace(day=1)
        filters["date_range"] = [
            first_day_this_month.strftime("%Y-%m-%d"),
            current_date.strftime("%Y-%m-%d")
        ]
        logger.info(f"📅 THIS MONTH DETECTED: {first_day_this_month.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
    
    # Enhanced unique masterid patterns - ADD THESE
    # unique_masterid_patterns = [
    #   r'give.*?error.*?logs?.*?writebackfail.*?today.*?only.*?unique.*?masterids?',  # Your exact query
    # r'error.*?logs?.*?writebackfail.*?unique.*?masterids?',                        # WritebackFail error logs unique masterids
    # r'give.*?error.*?logs?.*?only.*?unique.*?masterids?',                          # Error logs only unique masterids
    # r'show.*?unique.*?masterids?.*?for.*?(\w+)',                                   # Show unique masterids for WritebackFail
    # r'unique.*?masterids?.*?(\w+).*?error',                                        # Unique masterids WritebackFail error
    # r'(\w+).*?error.*?logs?.*?unique.*?masterids?'
    # ]
    
    # for pattern in unique_masterid_patterns:
    #     match = re.search(pattern, prompt, re.IGNORECASE)
    #     if match:
    #       keyword = None
    #     if match.lastindex:
    #         keyword = match.group(1)

    #               # If no keyword captured, extract from prompt
    #     if not keyword:
    #         keyword_match = re.search(r'(writebackfail|writeback|fail)', prompt, re.IGNORECASE)
    #         if keyword_match:
    #             keyword = keyword_match.group(1)

    #         filters["is_unique_query"] = True
    #         filters["unique_field"] = "masterID"
    #         filters["keyword"] = keyword
    #         filters["logName"] = "error"  # Always error logs for this pattern
            
    #         logger.info(f"🔍 UNIQUE MASTERID PATTERN DETECTED: keyword='{keyword}', logName='error'")
    #         break
    
    # # Field:value extraction
    # field_patterns = re.findall(r'(\w+)[:=]([\w\.\-]+)', prompt)
    
    # field_mapping = {
    #     'masterid': 'masterID',
    #     'appversion': 'appVersion', 
    #     'eventtype': 'eventType',
    #     'eventname': 'eventName',
    #     'modulename': 'moduleName',
    #     'tid': 'TID',
    #     'custid': 'custID',
    #     'pmsname': 'pMSName',
    #     'pmsversion': 'pMSVersion',
    #     'sikkausername': 'sikkaUserName',
    #     'machinename': 'machineName',
    #     'osversion': 'oSVersion',
    #     'osusername': 'oSUserName',
    #     'machineguid': 'machineGUID',
    #     'merchantaccountid': 'merchantAccountID',
    #     'isqamode': 'isQAMode',
    #     'transactionamount': 'transactionAmount',
    #     'paymentprocessor': 'paymentProcessor',
    #     'responsecode': 'responseCode',
    #     'logname': 'logName'
    # }

    
    
    # for field, value in field_patterns:
    #     field_lower = field.lower()
    #     if field_lower in field_mapping:
    #         filters[field_mapping[field_lower]] = value
    
    # # Aggregation detection
    # aggregation_keywords = ['unique', 'all', 'show me all', 'group by', 'distinct', 'by unique']
    # field_keywords = ['appversion', 'eventname', 'masterid', 'machinename']
    
    # if any(keyword in prompt_lower for keyword in aggregation_keywords):
    #     mentioned_fields = [field for field in field_keywords if field in prompt_lower]
    #     if len(mentioned_fields) >= 2:
    #         filters["is_aggregation"] = True
    #         filters["agg_fields"] = mentioned_fields
            
    #         # Extract eventName value if specified
    #         eventname_match = re.search(r'eventname[:=]?\s*([\w\(\)]+)', prompt, re.IGNORECASE)
    #         if eventname_match:
    #             filters["eventName"] = eventname_match.group(1)


#new code

def parse_natural_language_query(prompt: str) -> Dict:
    """Parse natural language queries - SAFE VERSION"""
    filters = {}
    prompt_lower = prompt.lower()
    current_date = datetime.now()
    
    logger.info(f"🔍 PARSING: {prompt}")

    # ✅ DATE DETECTION (your existing working code)
    if "today" in prompt_lower:
        today = current_date.strftime("%Y-%m-%d")
        filters["date"] = today
        logger.info(f"📅 TODAY DETECTED: {today}")
    elif "yesterday" in prompt_lower:
        yesterday = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        filters["date"] = yesterday
        logger.info(f"📅 YESTERDAY DETECTED: {yesterday}")
    elif "tomorrow" in prompt_lower:
        tomorrow = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        filters["date"] = tomorrow
        logger.info(f"📅 TOMORROW DETECTED: {tomorrow}")
    elif "last week" in prompt_lower:
        days_since_monday = current_date.weekday()
        last_monday = current_date - timedelta(days=days_since_monday + 7)
        last_sunday = last_monday + timedelta(days=6)
        filters["date_range"] = [
            last_monday.strftime("%Y-%m-%d"),
            last_sunday.strftime("%Y-%m-%d")
        ]
        logger.info(f"📅 LAST WEEK DETECTED: {last_monday.strftime('%Y-%m-%d')} to {last_sunday.strftime('%Y-%m-%d')}")

    # ✅ SAFE UNIQUE MASTERID DETECTION
    if 'unique' in prompt_lower and 'masterid' in prompt_lower and 'error' in prompt_lower:
        keyword_found = None
        for keyword in ['writebackfail', 'writeback', 'fail']:
            if keyword in prompt_lower:
                keyword_found = 'WritebackFail' if keyword == 'writebackfail' else keyword
                break
        
        if keyword_found:
            filters["is_unique_query"] = True
            filters["unique_field"] = "masterID"
            filters["keyword"] = keyword_found
            filters["logName"] = "error"
            logger.info(f"🔍 UNIQUE MASTERID DETECTED: keyword='{keyword_found}'")

    # ✅ SAFE FIELD:VALUE EXTRACTION
    try:
        field_mapping = {
            'masterid': 'masterID', 'appversion': 'appVersion', 'eventtype': 'eventType',
            'eventname': 'eventName', 'modulename': 'moduleName', 'tid': 'TID',
            'custid': 'custID', 'pmsname': 'pMSName', 'pmsversion': 'pMSVersion',
            'sikkausername': 'sikkaUserName', 'machinename': 'machineName',
            'osversion': 'oSVersion', 'osusername': 'oSUserName', 'machineguid': 'machineGUID',
            'merchantaccountid': 'merchantAccountID', 'isqamode': 'isQAMode',
            'transactionamount': 'transactionAmount', 'paymentprocessor': 'paymentProcessor',
            'responsecode': 'responseCode', 'logname': 'logName'
        }
        
        field_patterns = re.findall(r'(\w+)[:=]([\w\.\-]+)', prompt)
        for field, value in field_patterns:
            field_lower = field.lower()
            if field_lower in field_mapping:
                filters[field_mapping[field_lower]] = value
    except Exception:
        pass

    # ✅ SAFE AGGREGATION DETECTION
    aggregation_keywords = ['unique', 'all', 'show me all', 'group by', 'distinct', 'by unique']
    field_keywords = ['appversion', 'eventname', 'masterid', 'machinename']
    
    if any(keyword in prompt_lower for keyword in aggregation_keywords):
        mentioned_fields = [field for field in field_keywords if field in prompt_lower]
        if len(mentioned_fields) >= 2:
            filters["is_aggregation"] = True
            filters["agg_fields"] = mentioned_fields
            
            try:
                eventname_match = re.search(r'eventname[:=]?\s*([\w\(\)]+)', prompt, re.IGNORECASE)
                if eventname_match:
                    filters["eventName"] = eventname_match.group(1)
            except Exception:
                pass

    # ✅ Continue with your existing working code for dates, master IDs, etc.
    # (Keep all your existing code that was working)

    logger.info(f"🔍 FILTERS: {filters}")
    return filters

 # Enhanced date parsing - Handle date ranges FIRST
    if "date" not in filters and "date_range" not in filters:
        # Enhanced date range patterns
        date_range_patterns = [
            r'between\s+(.+?)\s+(?:and|to)\s+(.+?)(?:\s|$)',         
            r'from\s+(.+?)\s+to\s+(.+?)(?:\s|$)',                  
            r'(.+?)\s+to\s+(.+?)(?:\s|$)',                          
        ]
        
        # Try to parse date ranges first
        for pattern in date_range_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                start_str = match.group(1).strip()
                end_str = match.group(2).strip()
                
                # Clean date strings (remove common words)
                start_str = re.sub(r'\b(error|log|logs|from|for|masterid)\b', '', start_str, flags=re.IGNORECASE).strip()
                end_str = re.sub(r'\b(error|log|logs|from|for|masterid)\b', '', end_str, flags=re.IGNORECASE).strip()
                
                try:
                    start_date = dateparser.parse(start_str, settings={
                        'PREFER_DAY_OF_MONTH': 'first',
                        'DATE_ORDER': 'DMY',
                        'TIMEZONE': 'UTC'
                    })
                    end_date = dateparser.parse(end_str, settings={
                        'PREFER_DAY_OF_MONTH': 'first', 
                        'DATE_ORDER': 'DMY',
                        'TIMEZONE': 'UTC'
                    })
                    
                    if start_date and end_date:
                        # Set to noon UTC to avoid timezone shifts
                        start_date = start_date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=tz.UTC)
                        end_date = end_date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=tz.UTC)
                        
                        filters["date_range"] = [
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d")
                        ]
                        logger.info(f"📅 DATE RANGE DETECTED: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to parse date range: {start_str} to {end_str}")
                    continue
    
    # Your existing single date parsing continues here...
    # Handle relative dates (your existing logic)
    current_date = datetime.now()
    if "yesterday" in prompt_lower:
        yesterday = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        filters["date"] = yesterday


    # ISO timestamp parsing (2025-07-09T15:49:19.999Z format)
    iso_pattern = re.search(r'\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\b', prompt)
    if iso_pattern:
        iso_time = iso_pattern.group(1)
        try:
            dt = datetime.strptime(iso_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            start_time = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            end_dt = dt + timedelta(milliseconds=1)
            end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            filters["exact_eventTime"] = {"start": start_time, "end": end_time}
        except:
            pass
    
    # Exact eventTime parsing (Jul 9, 2025 @ 17:37:48.756 format)
    exact_time_pattern = re.search(r'(\w{3}\s+\d{1,2},\s+\d{4}\s+@\s+\d{2}:\d{2}:\d{2}\.\d{3})', prompt)
    if exact_time_pattern:
        time_str = exact_time_pattern.group(1)
        try:
            normalized_str = time_str.replace(' @ ', ' ')
            dt = dateutil.parser.parse(normalized_str)
            start_time = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            end_dt = dt + timedelta(milliseconds=1)
            end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            filters["exact_eventTime"] = {"start": start_time, "end": end_time}
        except:
            pass
    
    # "masterid D48662" pattern parsing
    masterid_space_pattern = re.search(r'masterid\s+([A-Z][0-9]{5})', prompt, re.IGNORECASE)
    if masterid_space_pattern:
        filters["masterID"] = masterid_space_pattern.group(1)
    
    # Master ID extraction
    if "masterID" not in filters:
        masterid_match = re.search(r'\b([A-Z][0-9]{5})\b', prompt)
        if masterid_match:
            filters["masterID"] = masterid_match.group(1)
    
    # Log type extraction
    if "logName" not in filters:
        log_types = {
            "error": ["error", "fail", "exception", "crash"],
            "access": ["access", "request", "visit", "hit"],
            "info": ["info", "information"],
            "debug": ["debug", "trace"],
            "warning": ["warning", "warn", "alert"],
            "critical": ["critical", "fatal", "severe"]
        }
        
        for log_type, keywords in log_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                filters["logName"] = log_type
                break
    
 

# REPLACE the date parsing section with this:

    # Enhanced flexible date parsing with timezone fix
    datetime_patterns = [
        r'log\s+(.*?)\s+masterid',
        r'error\s+(.*?)\s+masterid',  
        r'access\s+(.*?)\s+masterid',
        r'logs?\s+(.*?)\s+masterid',
        r'on\s+([\w\s,\-\:]+)',
        r'from\s+([\w\s,\-\:]+)\s+to\s+([\w\s,\-\:]+)',
    ]
    
    for pattern in datetime_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            if 'from' in pattern and 'to' in pattern:
                # Handle date ranges
                start_str = match.group(1).strip()
                end_str = match.group(2).strip()
                try:
                    start_dt = date_parser.parse(start_str, fuzzy=True)
                    end_dt = date_parser.parse(end_str, fuzzy=True)
                    
                    # Set to noon UTC to avoid timezone shifts
                    start_dt = start_dt.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=tz.UTC)
                    end_dt = end_dt.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=tz.UTC)
                    
                    filters["date_range"] = [
                        start_dt.strftime("%Y-%m-%d"),
                        end_dt.strftime("%Y-%m-%d")
                    ]
                    break
                except (ValueError, TypeError):
                    continue
            else:
                # Handle single dates
                datetime_text = match.group(1).strip()
                skip_words = ['today', 'yesterday', 'last week', 'writebackfail', 'timeout', 'failed']
                if datetime_text.lower() in skip_words or len(datetime_text) < 3:
                    continue

                try:
                    # FIXED: Use timezone-aware parsing
                    dt = date_parser.parse(datetime_text, fuzzy=True)
                    # Set to noon UTC to prevent date shifts
                    dt = dt.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=tz.UTC)
                    filters["date"] = dt.strftime("%Y-%m-%d")
                    logger.info(f"📅 DATE PARSED: '{datetime_text}' → {dt.strftime('%Y-%m-%d')}")
                    break
                except (ValueError, TypeError):
                    continue


    # Response code extraction
    if "responseCode" not in filters:
        status_codes = re.findall(r'\b(2\d{2}|3\d{2}|4\d{2}|5\d{2})\b', prompt)
        if status_codes:
            filters["responseCode"] = status_codes[0]
    
    # Keyword extraction
    if "containing" in prompt_lower:
        keyword_match = re.search(r'containing\s+(\w+)', prompt_lower)
        if keyword_match:
            filters["keyword"] = keyword_match.group(1)
    
    return filters

def build_unique_value_query(filters: Dict) -> Dict:
    """Build query to find unique masterIDs with keyword and logName filtering"""
    
    must_clauses = []
    
    # Add keyword filter (search in details field)
    if filters.get("keyword"):
        must_clauses.append({
            "match": {"details": filters["keyword"]}
        })
    
    # Add logName filter if specified
    if filters.get("logName"):
        must_clauses.append({
            "term": {"logName.keyword": filters["logName"]}
        })
    
    # Add date filter for "today"
    if filters.get("date"):
        date_str = filters["date"]
        must_clauses.append({
            "range": {
                "eventTime": {
                    "gte": f"{date_str}||/d",
                    "lt": f"{date_str}||+1d/d",
                    "time_zone": "+05:30"  # IST timezone
                }
            }
        })

    # Build query
    base_query = {"match_all": {}} if not must_clauses else {"bool": {"must": must_clauses}}
    
    return {
        "size": 0,
        "track_total_hits": True,
        "query": base_query,
        "aggs": {
            "unique_masterids": {
                "terms": {
                    "field": "masterID.keyword",
                    "size": 1000,
                    "order": {"_key": "asc"},  # ✅ CONSISTENT: Alphabetical order
                    "min_doc_count": 1  # ✅ FILTER: Only include IDs with at 
                }
            },
"aggs": {
                    "sample_doc": {
                        "top_hits": {
                            "size": 1,  # ✅ ONE DOC: Get one sample document per masterID
                            "_source": ["masterID", "eventTime", "logName", "details"],
                            "sort": [{"eventTime": {"order": "desc"}}]
                        }
                    }
                }
        }
    }

def process_unique_value_response(response: Dict, filters: Dict) -> list:
    """Process unique masterID aggregation response with enhanced deduplication"""
    
    results = []
    aggs = response.get("aggregations", {})
    unique_masterids_seen = set()  # ✅ TRACK: Prevent duplicates
    
    for bucket in aggs.get("unique_masterids", {}).get("buckets", []):
        masterid = bucket["key"]
        # ✅ SKIP: If we've already seen this masterID
        if masterid in unique_masterids_seen:
            continue

        
        unique_masterids_seen.add(masterid)

         # Get sample document for additional details
        sample_doc = None
        if "sample_doc" in bucket:
            hits = bucket["sample_doc"].get("hits", {}).get("hits", [])
            if hits:
                sample_doc = hits[0].get("_source", {})

            results.append({
                "unique_masterid": bucket["key"],
                "doc_count": bucket["doc_count"],
            "keyword": filters.get("keyword", ""),
            "logName": filters.get("logName", ""),
            "date": filters.get("date", ""),
            "sample_time": sample_doc.get("eventTime", "") if sample_doc else "",
            "type": "unique_masterid"
        })
    
    return results
@app.get("/debug/unique-masterids")
async def debug_unique_masterids():
    """Debug endpoint to test unique masterID aggregation"""
    
    if not es:
        return JSONResponse({"error": "OpenSearch not connected"}, status_code=503)
    
    # Test query for WritebackFail error logs today
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    
    test_query = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"match": {"details": "WritebackFail"}},
                    {"term": {"logName.keyword": "error"}},
                    {
                        "range": {
                            "eventTime": {
                                "gte": f"{today}||/d",
                                "lt": f"{today}||+1d/d",
                                "time_zone": "+05:30"
                            }
                        }
                    }
                ]
            }
        },
        "aggs": {
            "unique_masterids": {
                "terms": {
                    "field": "masterID.keyword",
                    "size": 1000,
                    "order": {"_key": "asc"}
                }
            }
        }
    }
    
    try:
        response = es.search(index=ES_INDEX, body=test_query)
        buckets = response.get("aggregations", {}).get("unique_masterids", {}).get("buckets", [])
        
        return JSONResponse({
            "query": test_query,
            "total_hits": response.get("hits", {}).get("total", {}),
            "unique_count": len(buckets),
            "unique_masterids": [bucket["key"] for bucket in buckets],
            "buckets": buckets[:10],  # First 10 buckets for inspection
            "took": response.get("took", 0)
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def build_simple_aggregation_query(filters: Dict) -> Dict:
    """Composite aggregation query compatible with older OpenSearch"""
    
    query_filters = []
    
    # Add eventName filter if specified
    if filters.get("eventName"):
        query_filters.append({
            "term": {"eventName.keyword": filters["eventName"]}
        })
    
    # Base query
    base_query = {
        "match_all": {}
    } if not query_filters else {
        "bool": {"filter": query_filters}
    }

    return {
        "size": 0,
        "track_total_hits": True,
        "query": base_query,
        "aggs": {
            "composite_buckets": {
                "composite": {
                    "size": 100,
                    "sources": [
                        {"masterID": {"terms": {"field": "masterID.keyword"}}},
                        {"machineName": {"terms": {"field": "machineName.keyword"}}}
                    ]
                },
                "aggs": {
                    "unique_appVersions": {
                        "terms": {
                            "field": "appVersion.keyword",
                            "size": 10
                        }
                    },
                    "unique_eventNames": {
                        "terms": {
                            "field": "eventName.keyword",
                            "size": 10
                        }
                    }
                }
            }
        }
    }

def process_simple_aggregation_response(response: Dict, filters: Dict) -> list:
    """Process multi_terms aggregation response"""
    
    results = []
    aggs = response.get("aggregations", {})
    
    for bucket in aggs.get("composite_buckets", {}).get("buckets", []):
        # Extract masterID and machineName from the key
        master_id = bucket["key"]["masterID"]
        machine_name = bucket["key"]["machineName"]
        
        # Get unique app versions
        app_versions = [b["key"] for b in bucket.get("unique_appVersions", {}).get("buckets", [])]
        event_names = [b["key"] for b in bucket.get("unique_eventNames", {}).get("buckets", [])]
        results.append({
            "masterID": master_id,
            "machineName": machine_name,
            "appVersions": app_versions,
            "appVersionCount": len(app_versions),
            "eventNameCount": len(event_names),
            "docCount": bucket["doc_count"],
            "type": "aggregation",
            "eventName": filters.get("eventName", "")
        })
    
    return results

def build_opensearch_query(filters: Dict, max_results: int = 20) -> Dict:
    """Build OpenSearch query from parsed filters with enhanced field support"""
    
    must_clauses = []
    filter_clauses = []
    
    # Enhanced masterID filtering with multiple field variants
    if filters.get("masterID"):
        masterid_value = filters["masterID"]
        masterid_should = [
            {"term": {"masterID.keyword": masterid_value}},
            {"term": {"masterID": masterid_value}},
            {"term": {"masterId.keyword": masterid_value}},
            {"term": {"masterId": masterid_value}},
            {"match": {"masterID": masterid_value}}
        ]
        must_clauses.append({
            "bool": {
                "should": masterid_should,
                "minimum_should_match": 1
            }
        })
    
    # Enhanced logName filtering with multiple field variants
    if filters.get("logName"):
        logname_value = filters["logName"]
        logname_should = [
            {"term": {"logName.keyword": logname_value}},
            {"term": {"logName": logname_value}},
            {"term": {"log_name.keyword": logname_value}},
            {"term": {"level.keyword": logname_value}},
            {"match": {"logName": logname_value}}
        ]
        must_clauses.append({
            "bool": {
                "should": logname_should,
                "minimum_should_match": 1
            }
        })
    
    # Enhanced exact eventTime filtering
    if filters.get("exact_eventTime"):
        time_should = [
            {"range": {"eventTime": {"gte": filters["exact_eventTime"]["start"], "lt": filters["exact_eventTime"]["end"]}}},
            {"range": {"@timestamp": {"gte": filters["exact_eventTime"]["start"], "lt": filters["exact_eventTime"]["end"]}}},
            {"range": {"timestamp": {"gte": filters["exact_eventTime"]["start"], "lt": filters["exact_eventTime"]["end"]}}}
        ]
        filter_clauses.append({
            "bool": {
                "should": time_should,
                "minimum_should_match": 1
            }
        })
    
    # Handle other extracted fields
    field_searches = ['appVersion', 'eventType', 'eventName', 'moduleName', 'TID', 'custID', 
                     'merchantAccountID', 'transactionAmount', 'paymentProcessor', 'machineName']
    for field in field_searches:
        if field in filters:
            value = filters[field]
            should_clauses = [
                {"term": {f"{field}.keyword": value}},
                {"term": {field: value}},
                {"match": {field: value}}
            ]
            must_clauses.append({
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            })
    
    # Handle date filtering
    if filters.get("date"):
        date_str = filters["date"]
        filter_clauses.append({
            "range": {
                "eventTime": {
             "gte": f"{date_str}||/d",      # Start of day
                "lt": f"{date_str}||+1d/d",    # Start of next day
                "time_zone": "+05:30"          # ✅ ADD: IST timezone
                }
            }
        })
    elif filters.get("date_range"):
        start_date, end_date = filters["date_range"]
    # # Calculate exact end of the end date
     # Calculate next day boundary for exclusion
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        next_day_boundary = end_dt.strftime("%Y-%m-%dT00:00:00Z")
        
        filter_clauses.append({
            "range": {
                "eventTime": {
 "gte": f"{start_date}T00:00:00",
              "gte": f"{start_date}||/d",
                "lt": f"{end_date}||+1d/d",
                "time_zone": "+05:30"  # ✅ ADD: IST timezone
                }
            }
        })
    
    # Handle keyword search
    if filters.get("keyword"):
        keyword = filters["keyword"]
        must_clauses.append({
            "multi_match": {
                "query": keyword,
                "fields": ["details", "message", "description"]
            }
        })
    
    # Build final query
    if not must_clauses and not filter_clauses:
        query = {"match_all": {}}
    else:
        query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
    
    return {
        "query": query,
        "sort": [
            {"_score": {"order": "desc"}},
            {"eventTime": {"order": "desc", "unmapped_type": "date"}},
            {"@timestamp": {"order": "desc", "unmapped_type": "date"}}
        ],
        "size": min(max_results, 50),
        "track_total_hits": True,
        "_source": {
            "includes": [
                "masterID", "logName", "eventType", "eventName", "moduleName",
                "details", "custID", "pMSName", "pMSVersion", "sikkaUserName",
                "eventTime", "machineName", "appVersion", "oSVersion", 
                "oSUserName", "machineGUID", "merchantAccountID", "isQAMode",
                "transactionAmount", "paymentProcessor"
            ]
        }
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

async def serve_chat_interface(request: Request):
    """Serve the chat interface"""
    try:
        return templates.TemplateResponse("chat.html", {"request": request})
    except Exception as e:
        logger.error(f"Template rendering failed: {e}")
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kibana AI Assistant - Setup Required</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8fafc; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .success {{ color: #059669; font-size: 1.2rem; margin-bottom: 20px; }}
                .error {{ color: #dc2626; margin-bottom: 20px; }}
                .steps {{ background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .step {{ margin: 10px 0; padding: 10px; background: white; border-radius: 6px; }}
                h1 {{ color: #1f2937; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Kibana AI Assistant</h1>
                <div class="success">
                    ✅ Backend is running successfully!<br>
                    OpenSearch connected with 66+ million documents ready to search.
                </div>
                <div class="error">
                    ❌ Chat interface template missing<br>
                    Create <code>templates/chat.html</code> file with the provided HTML code.
                </div>
                <div class="steps">
                    <div class="step"><strong>Step 1:</strong> Create <code>templates</code> folder</div>
                    <div class="step"><strong>Step 2:</strong> Save chat.html in templates folder</div>
                    <div class="step"><strong>Step 3:</strong> Restart server</div>
                </div>
                <h2>Available Endpoints</h2>
                <ul>
                    <li><a href="/docs">📚 API Documentation</a></li>
                    <li><a href="/health">🏥 Health Check</a></li>
                </ul>
            </div>
        </body>
        </html>
        """, status_code=404)

@app.post("/query")
async def query_logs(request: QueryRequest):
    """Enhanced query endpoint with pagination support"""
    
    if not es:
        return JSONResponse({
            "error": "I'm having trouble connecting to the log database 😔",
            "suggestion": "Please check the OpenSearch connection"
        }, status_code=503)
    
    prompt = request.prompt.strip()
    if not prompt:
        return JSONResponse({
            "error": "Please ask me something about your logs! 😊",
            "suggestions": [
                "Try: 'Show me error logs from yesterday'",
                "Or: 'Find 500 errors in last 3 days'"
            ]
        })
    
    # ✅ CRITICAL: Initialize filters BEFORE try block
    filters = {}
    
    try:
        # Get pagination parameters
        offset = getattr(request, 'offset', 0)
        page_size = getattr(request, 'page_size', 20)
        
        # Parse natural language query
        filters = parse_natural_language_query(prompt)
        
        # PRIORITY 1: Handle unique value queries FIRST
        if filters.get("is_unique_query"):
            query_body = build_unique_value_query(filters)
            response = es.search(index=ES_INDEX, body=query_body)
            results = process_unique_value_response(response, filters)
            
            # Enhanced interpretation message
            keyword = filters.get("keyword", "")
            log_type = filters.get("logName", "")
            interpretation = f"Found {len(results)} unique masterIDs in {log_type} logs containing '{keyword}'"
            
            return JSONResponse({
                "results": results,
                "total_found": len(results),
                "query_interpretation": interpretation,
                "is_unique_query": True,
                "filters_applied": filters,
                "took": response.get("took", 0)
            })
        
        # PRIORITY 2: Handle aggregation queries SECOND
        if filters.get("is_aggregation"):
            query_body = build_simple_aggregation_query(filters)
            response = es.search(index=ES_INDEX, body=query_body)
            results = process_simple_aggregation_response(response, filters)
            
            return JSONResponse({
                "results": results,
                "total_found": len(results),
                "query_interpretation": f"Found {len(results)} unique combinations",
                "is_aggregation": True,
                "filters_applied": filters,
                "took": response.get("took", 0)
            })
        
        # PRIORITY 3: Regular search queries
        # Build OpenSearch query with pagination
        index = request.index or ES_INDEX
        query_body = build_opensearch_query(filters, page_size)
        
        # Add pagination to query
        query_body["from"] = offset
        query_body["size"] = page_size

        # DEBUG: Log the generated query
        logger.info(f"Generated query: {json.dumps(query_body, indent=2)}")
        logger.info(f"Extracted filters: {filters}")
        
        # Execute search
        response = es.search(index=index, body=query_body)
        hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {})
        
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "masterID": source.get("masterID", ""),
                "logName": source.get("logName", ""),
                "eventType": source.get("eventType", ""),
                "eventName": source.get("eventName", ""),
                "moduleName": source.get("moduleName", ""),
                "details": source.get("details", ""),
                "custID": source.get("custID", ""),
                "pMSName": source.get("pMSName", ""),
                "pMSVersion": source.get("pMSVersion", ""),
                "sikkaUserName": source.get("sikkaUserName", ""),
                "eventTime": source.get("eventTime", ""),
                "machineName": source.get("machineName", ""),
                "appVersion": source.get("appVersion", ""),
                "oSVersion": source.get("oSVersion", ""),
                "oSUserName": source.get("oSUserName", ""),
                "machineGUID": source.get("machineGUID", ""),
                "merchantAccountID": source.get("merchantAccountID", ""),
                "isQAMode": source.get("isQAMode", ""),
                "transactionAmount": source.get("transactionAmount", ""),
                "paymentProcessor": source.get("paymentProcessor", ""),
                "_score": hit.get("_score", 0)
            })
        
        # Generate friendly response
        result_count = len(results)
        
        if result_count == 0:
            interpretation = f"I searched for logs matching '{prompt}' but didn't find any results."
        elif result_count == 1:
            interpretation = f"Perfect! I found 1 log entry matching '{prompt}'"
        else:
            interpretation = f"Great! I found {result_count} log entries matching '{prompt}'"
        
        return JSONResponse({
            "results": results,
            "total_found": result_count,
            "total_available": total_count,
            "query_interpretation": interpretation,
            "filters_applied": filters,
            "took": response.get("took", 0),
            "offset": offset,
            "has_more": (offset + result_count) < total_count
        })
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return JSONResponse({
            "error": f"I encountered an error while searching: {str(e)}",
            "suggestion": "Try rephrasing your question"
        }, status_code=500)

@app.post("/query/more")
async def query_more_logs(request: PaginatedQueryRequest):
    """Get more results for pagination"""
    return await query_logs(request)

@app.get("/health")  
async def health_check():
    """System health check"""
    
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }
    
    # Check OpenSearch
    if es:
        try:
            info = es.info()
            status["opensearch"] = {
                "status": "connected",
                "cluster_name": info.get("cluster_name"),
                "version": info.get("version", {}).get("number"),
                "host": ES_HOST,
                "index": ES_INDEX
            }
            
            # Check index
            exists = es.indices.exists(index=ES_INDEX)
            status["opensearch"]["index_exists"] = exists
            
            if exists:
                stats = es.indices.stats(index=ES_INDEX)
                doc_count = stats['indices'][ES_INDEX]['total']['docs']['count']
                status["opensearch"]["document_count"] = doc_count
                
        except Exception as e:
            status["opensearch"] = {"status": "error", "error": str(e)}
            status["status"] = "degraded"
    else:
        status["opensearch"] = {"status": "not_connected"}
        status["status"] = "degraded"
    
    return JSONResponse(status)

@app.get("/api/info")
async def api_info():
    """API information"""
    return JSONResponse({
        "name": "Kibana AI Assistant",
        "version": "2.0.0",
        "description": "AI-powered chat interface for OpenSearch log analysis",
        "features": [
            "Natural language query processing",
            "Real-time log search",
            "Conversational chat interface",
            "Advanced filtering and parsing"
        ],
        "endpoints": {
            "GET /": "Chat interface",
            "POST /query": "Natural language query processing",
            "GET /health": "System health check",
            "GET /docs": "API documentation"
        }
    })

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Kibana AI Assistant...")
    print(f"📊 OpenSearch: {ES_HOST}")
    print(f"📋 Index: {ES_INDEX}")
    print(f"🗣️ Interface: Chat")
    print(f"🔧 Debug Mode: {DEBUG_MODE}")
    print("=" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=DEBUG_MODE, log_level="info")
