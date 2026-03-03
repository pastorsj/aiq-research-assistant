// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Data Sources Type Definitions
 *
 * Type definitions for data sources returned by the backend API.
 * Data sources are fetched dynamically from GET /v1/data_sources.
 */

/** Category types for organizing data sources */
export type DataSourceCategory = 'web' | 'enterprise' | 'storage' | 'collaboration'

/** Data source configuration interface */
export interface DataSource {
  /** Unique identifier matching backend source IDs */
  id: string
  /** Display name for the source */
  name: string
  /** Brief description of the source */
  description: string
  /** Category for grouping/filtering */
  category: DataSourceCategory
  /** Whether the source is enabled by default */
  defaultEnabled: boolean
}

/**
 * Default data source ID that works without authentication.
 * Web search uses backend API keys (Tavily), not user tokens.
 */
export const WEB_SEARCH_SOURCE_ID = 'web_search'

/**
 * Enterprise knowledge source ID — pre-indexed collections that don't require
 * user authentication (no file uploads, no per-user tokens).
 */
export const ENTERPRISE_KNOWLEDGE_SOURCE_ID = 'enterprise_knowledge'

/**
 * Set of source IDs that work without user authentication.
 * These sources use backend-managed API keys or pre-indexed data.
 */
const NO_AUTH_SOURCE_IDS = new Set([WEB_SEARCH_SOURCE_ID, ENTERPRISE_KNOWLEDGE_SOURCE_ID])

/** Check whether a data source requires user authentication. */
export const doesDataSourceNeedAuthentication = (sourceId: string): boolean =>
  !NO_AUTH_SOURCE_IDS.has(sourceId)
