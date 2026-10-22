<script lang="ts">
	import { Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		ChatFormActionModels,
		ChatFormActionRecord,
		ChatFormActionSubmit,
		ChatFormContextGauge
	} from '$lib/components/app';
	import { FileTypeCategory, MessageRole } from '$lib/enums';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { activeMessages, conversationsStore } from '$lib/stores/conversations.svelte';
	import {
		activeProcessingState,
		isChatStreaming,
		isLoading as chatIsLoading
	} from '$lib/stores/chat.svelte';
	import { getFileTypeCategory } from '$lib/utils';
	import { page } from '$app/state';

	interface Props {
		canSend?: boolean;
		canSubmit?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isReasoning?: boolean;
		isRecording?: boolean;
		showAddButton?: boolean;
		showModelSelector?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: () => void;
		onMicClick?: () => void;
		onStop?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		canSend = false,
		canSubmit = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isReasoning = false,
		isRecording = false,
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = [],
		onFileUpload,
		onMicClick,
		onStop,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpResourcesClick
	}: Props = $props();

	let currentConfig = $derived(config());

	let hasMcpPromptsSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

		return mcpStore.hasPromptsCapability(perChatOverrides);
	});

	let hasMcpResourcesSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

		return mcpStore.hasResourcesCapability(perChatOverrides);
	});

	let hasAudioModality = $state(false);
	let hasVideoModality = $state(false);
	let hasVisionModality = $state(false);
	let hasModelSelected = $state(false);
	let isSelectedModelInCache = $state(true);
	let submitTooltip = $state('');

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !canSubmit && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let selectorModelRef: ChatFormActionModels | undefined = $state(undefined);

	export function openModelSelector() {
		selectorModelRef?.open();
	}

	let hasProcessedTokens = $derived.by(() => {
		if (!page.params.id) return false;

		const messages = activeMessages() as DatabaseMessage[];
		let totalHistoricalTokens = 0;
		for (const m of messages) {
			if (m.role !== MessageRole.ASSISTANT) continue;
			const timings = m.timings;
			if (!timings) continue;
			const agenticLlm = timings.agentic?.llm;
			if (agenticLlm?.prompt_n != null || agenticLlm?.predicted_n != null) {
				totalHistoricalTokens += (agenticLlm?.prompt_n ?? 0) + (agenticLlm?.predicted_n ?? 0);
			} else {
				totalHistoricalTokens += (timings.prompt_n ?? 0) + (timings.predicted_n ?? 0);
			}
		}
		if (totalHistoricalTokens > 0) return true;

		if (!chatIsLoading() && !isChatStreaming()) return false;

		const processingState = activeProcessingState();
		if (!processingState) return false;
		const livePromptTokens = Math.max(
			processingState.promptTokens ?? 0,
			processingState.promptProgress?.processed ?? 0
		);
		const liveOutputTokens = processingState.outputTokensUsed ?? 0;
		return livePromptTokens > 0 || liveOutputTokens > 0;
	});
</script>

<div
	class="flex w-full items-center gap-3 {className} justify-end"
	style="container-type: inline-size"
>
	<div class="flex items-center gap-1.5">
		{#if hasProcessedTokens}
			<ChatFormContextGauge />
		{/if}

		{#if showModelSelector}
			<ChatFormActionModels
				{disabled}
				bind:this={selectorModelRef}
				bind:hasAudioModality
				bind:hasVideoModality
				bind:hasVisionModality
				bind:hasModelSelected
				bind:isSelectedModelInCache
				bind:submitTooltip
				forceForegroundText
				useGlobalSelection
			/>
		{/if}
	</div>

	{#if isLoading && !canSubmit}
		<Button
			type="button"
			variant="secondary"
			onclick={onStop}
			class="group h-8 w-8 rounded-full p-0 hover:bg-destructive/10!"
		>
			<span class="sr-only">Stop</span>

			<Square
				class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
			/>
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && (showModelSelector ? hasModelSelected && isSelectedModelInCache : true)}
			{disabled}
			tooltipLabel={submitTooltip}
			showErrorState={showModelSelector && hasModelSelected && !isSelectedModelInCache}
		/>
	{/if}
</div>
