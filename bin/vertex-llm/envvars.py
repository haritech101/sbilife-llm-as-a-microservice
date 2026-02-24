class EnvVars:
    test_type = "TEST_TYPE"
    http_port_qa = "HTTP_PORT_QA"
    http_port_material = "HTTP_PORT_MATERIAL"
    vertex_ai_region = "VERTEX_AI_REGION"
    vertex_ai_project_id = "VERTEX_AI_PROJECT_ID"
    vertex_ai_model = "VERTEX_AI_MODEL"
    min_chunk_size = "MIN_CHUNK_SIZE"
    max_output_tokens = "MAX_OUTPUT_TOKENS"
    google_application_credentials = "GOOGLE_APPLICATION_CREDENTIALS"


class Defaults:
    test_type = "unit"  # or "integration"
    http_port_qa = "80"
    http_port_material = "81"
    vertex_ai_region = "us-central1"
    vertex_ai_model = "claude-sonnet-4"
    min_chunk_size = "4000"
    max_output_tokens = "8192"
