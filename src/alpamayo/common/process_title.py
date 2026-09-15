# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

logger = logging.getLogger(__name__)


def configure_process_title() -> str | None:
    """Set a short process title for torchrun workers when requested.

    Environment variables:
        PROCESS_TITLE: Base display name, e.g. "Stage1-2B".
        ALPAMAYO_PROCESS_TITLE: Alpamayo-specific alias for PROCESS_TITLE.
        PROCESS_TITLE_TEMPLATE: Optional format string. Available fields are
            {title}, {local_rank}, {rank}, and {world_size}.

    Returns:
        The title that was set, or None when process-title customization is disabled.
    """
    title = (
        os.environ.get("ALPAMAYO_PROCESS_TITLE")
        or os.environ.get("PROCESS_TITLE")
        or ""
    ).strip()
    if not title:
        return None

    local_rank = os.environ.get("LOCAL_RANK", "0")
    rank = os.environ.get("RANK", local_rank)
    world_size = os.environ.get("WORLD_SIZE", "1")
    template = os.environ.get("PROCESS_TITLE_TEMPLATE", "[{title}] worker-{local_rank}")
    fields = {
        "title": title,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
    }
    try:
        process_title = template.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        if local_rank == "0":
            logger.warning(
                "Invalid PROCESS_TITLE_TEMPLATE %r (%s); using the default template.",
                template,
                exc,
            )
        template = "[{title}] worker-{local_rank}"
        process_title = template.format(**fields)

    try:
        from setproctitle import setproctitle
    except ImportError:
        if local_rank == "0":
            logger.warning(
                "PROCESS_TITLE is set but setproctitle is not installed; "
                "process title was not changed."
            )
        return None

    setproctitle(process_title)
    if local_rank == "0":
        logger.info("Set process title template to: %s", template)
    return process_title
