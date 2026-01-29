/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://github.com/jd-opensource/xllm/blob/main/LICENSE
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef XLLM_VLM_API_H
#define XLLM_VLM_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "types.h"

/**
 * @brief Opaque handle to a Generative Recommendation (VLM) inference instance
 * This handle encapsulates all internal state of a VLM-specialized runtime,
 * including:
 * - Generative recommendation model weights (item embedding, ranking head)
 * - Device context (CUDA/NPU streams for batch inference)
 * - Generation cache (user behavior context, item candidate pool)
 * - Runtime config (recommendation-specific decoding strategy)
 * The handle MUST be created via xllm_vlm_create() and destroyed via
 * xllm_vlm_destroy() to prevent memory/device resource leaks.
 */
typedef struct XLLM_VLM_Handler XLLM_VLM_Handler;

/**
 * @brief Create a new Generative Recommendation (VLM) inference instance handle
 * This is the first function that must be called before using any other VLM
 * APIs.
 * @return Valid XLLM_VLM_Handler* on success; NULL if memory allocation fails
 * @see xllm_vlm_destroy
 */
XLLM_CAPI_EXPORT XLLM_VLM_Handler* xllm_vlm_create(void);

/**
 * @brief Destroy a Generative Recommendation (VLM) inference instance handle
 * and release resources Frees all memory allocated for the VLM instance,
 * including:
 * - Model weights (host/device memory for item embedding and ranking head)
 * - Runtime context (CUDA/NPU streams, compute graphs for batch recommendation)
 * - Generation cache (user behavior sequence, item candidate pool, attention
 * cache)
 * - Device resources (contexts, queues, memory pools for batch inference)
 * This function is idempotent—calling with NULL has no effect.
 * @param handler VLM inference instance handle (NULL = no operation)
 * @note Mandatory: Must be called to avoid memory/device resource leaks
 * @see xllm_vlm_create
 */
XLLM_CAPI_EXPORT void xllm_vlm_destroy(XLLM_VLM_Handler* handler);

/**
 * @brief Helper to initialize XLLM_InitOptions with VLM default values
 * Copies the predefined XLLM_INIT_VLM_OPTIONS_DEFAULT values into the target
 * init_options struct. Convenient alternative to manually setting each field,
 * ensuring consistency with VLM best practices.
 * @param init_options Pointer to XLLM_InitOptions to initialize (NULL = no-op)
 * @see XLLM_INIT_VLM_OPTIONS_DEFAULT, xllm_vlm_initialize
 */
XLLM_CAPI_EXPORT void xllm_vlm_init_options_default(
    XLLM_InitOptions* init_options);

/**
 * @brief Initialize the Generative Recommendation (VLM) model and runtime
 * environment Loads generative recommendation model weights from the specified
 * path, configures target devices, initializes compute contexts, and prepares
 * the recommendation inference runtime
 * @param handler Valid VLM inference instance handle (must not be NULL)
 * @param model_path Null-terminated string of the VLM model directory/file path
 *                   (supports .bin/.pth/.safetensors formats with ranking head)
 * @param devices Null-terminated string specifying target devices (format:
 *                "npu:0,1" (specific NPUs), "cuda:0" (single GPU), "auto"
 * (automatic selection))
 * @param init_options Advanced initialization options (NULL = use VLM defaults)
 * @return true if initialization succeeds; false on failure (see failure causes
 * below)
 * @par Failure Causes
 * - Invalid handler (NULL or already destroyed)
 * - Invalid model_path (non-existent, corrupted, or missing ranking head
 * weights)
 * - Invalid devices string (malformed format or unavailable devices)
 * - Model load error (mismatched VLM model architecture or embedding table
 * corruption)
 * - Device initialization failure (out of memory, driver error, insufficient
 * batch size)
 * @see xllm_vlm_init_options_default, XLLM_INIT_VLM_OPTIONS_DEFAULT,
 * xllm_vlm_create
 */
XLLM_CAPI_EXPORT bool xllm_vlm_initialize(XLLM_VLM_Handler* handler,
                                          const char* model_path,
                                          const char* devices,
                                          const XLLM_InitOptions* init_options);

/**
 * @brief Helper to initialize XLLM_RequestParams with VLM default values
 * Copies the predefined XLLM_VLM_REQUEST_PARAMS_DEFAULT values into the target
 * request_params struct.
 * @param request_params Pointer to XLLM_RequestParams to initialize (NULL =
 * no-op)
 * @see XLLM_VLM_REQUEST_PARAMS_DEFAULT, xllm_vlm_text_completions,
 * xllm_vlm_token_completions, xllm_vlm_chat_completions
 */
XLLM_CAPI_EXPORT void xllm_vlm_request_params_default(
    XLLM_RequestParams* request_params);

/**
 * @brief Generate generative recommendation chat completions from multi-turn
 * conversation history Generates personalized recommendation responses for a
 * multi-turn user-assistant conversation
 * @param handler Valid, initialized VLM inference instance handle (must not be
 * NULL)
 * @param model_id Null-terminated string of the loaded VLM model ID
 * @param messages Array of XLLM_ChatMessage structs (recommendation-focused
 * conversation history)
 * @param messages_count Number of messages in the messages array (must be ≥ 0)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use VLM defaults)
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 * @par Response Status Codes
 * - kSuccess: Valid chat recommendation response generated (check
 * response->choices[0].message for item list)
 * - kNotInitialized: Handler not initialized with xllm_vlm_initialize()
 * - kInvalidRequest: Invalid messages (NULL with count>0, empty role/content,
 * non-recommendation context)
 * - kTimeout: Generation exceeded timeout_ms
 * @warning Mandatory: Call xllm_vlm_free_response() to release response memory
 * @see xllm_vlm_request_params_default, XLLM_VLM_REQUEST_PARAMS_DEFAULT,
 * xllm_vlm_free_response
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_vlm_chat_completions(
    XLLM_VLM_Handler* handler,
    const char* model_id,
    const XLLM_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Free all dynamically allocated memory in a generative recommendation
 * XLLM_Response Releases all heap memory used by the VLM response struct
 * @param resp Pointer to XLLM_Response to free (NULL = no operation)
 * @warning Mandatory: Must be called after using VLM completions/chat
 * completions responses
 * @see xllm_vlm_text_completions, xllm_vlm_token_completions,
 * xllm_vlm_chat_completions
 */
XLLM_CAPI_EXPORT void xllm_vlm_free_response(XLLM_Response* resp);

#ifdef __cplusplus
}
#endif

#endif  // XLLM_VLM_API_H