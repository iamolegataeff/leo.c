/*
 * leo_jni.c — JNI bridge for Leo C inference on Android
 *
 * Wraps leo.c functions for Kotlin access.
 * GGUF loading, Gemma-3 forward pass, Zikharon memory — all via native code.
 *
 * (c) 2026 arianna method
 */

#include <jni.h>
#include <android/log.h>
#include <string.h>
#include <stdlib.h>

#define LOG_TAG "LeoJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/*
 * TODO: Include leo.c inference functions.
 * For beta, we provide stub implementations that will be replaced
 * when leo.c is adapted for Android (remove main(), add JNI entry points).
 *
 * The full implementation requires:
 * 1. leo.c compiled without main() (#ifdef LEO_JNI guards)
 * 2. GGUF mmap on Android (works with standard mmap)
 * 3. Zikharon load/save to app internal storage
 * 4. Thread-safe generation (single inference at a time)
 */

static int g_initialized = 0;
static char g_gguf_path[512] = {0};
static char g_mem_path[512] = {0};

JNIEXPORT jboolean JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeInit(
    JNIEnv *env, jobject thiz, jstring gguf_path, jstring mem_path)
{
    const char *gp = (*env)->GetStringUTFChars(env, gguf_path, NULL);
    const char *mp = (*env)->GetStringUTFChars(env, mem_path, NULL);
    snprintf(g_gguf_path, sizeof(g_gguf_path), "%s", gp);
    snprintf(g_mem_path, sizeof(g_mem_path), "%s", mp);
    (*env)->ReleaseStringUTFChars(env, gguf_path, gp);
    (*env)->ReleaseStringUTFChars(env, mem_path, mp);

    LOGI("Init: gguf=%s mem=%s", g_gguf_path, g_mem_path);

    /*
     * TODO: Call index_load(), zk_load() from leo.c
     * For beta stub: mark as initialized
     */
    g_initialized = 1;
    LOGI("Leo initialized (beta stub)");
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeGenerate(
    JNIEnv *env, jobject thiz, jstring prompt, jint max_tokens)
{
    if (!g_initialized) {
        return (*env)->NewStringUTF(env, "Leo is not ready yet...");
    }

    const char *p = (*env)->GetStringUTFChars(env, prompt, NULL);
    LOGI("Generate: prompt='%.50s...' max=%d", p, max_tokens);

    /*
     * TODO: Full inference pipeline:
     * 1. Wrap prompt in Gemma chat template
     * 2. Tokenize with SentencePiece from GGUF
     * 3. Prefill all prompt tokens
     * 4. Generate max_tokens with Dario field + Zikharon injection
     * 5. Decode tokens back to UTF-8
     *
     * For beta: return stub response showing prompt was received
     */
    char result[1024];
    snprintf(result, sizeof(result),
        "Leo hears: \"%.100s\" — inference engine loading...", p);

    (*env)->ReleaseStringUTFChars(env, prompt, p);
    return (*env)->NewStringUTF(env, result);
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeDream(JNIEnv *env, jobject thiz)
{
    if (!g_initialized) return (*env)->NewStringUTF(env, "...");

    /*
     * TODO: Dream cycle:
     * 1. Pick random anchor from Zikharon
     * 2. Generate 20 tokens from anchor tokens as seed
     * 3. Ingest into co-occurrence field
     * 4. Return dream text for widget display
     */
    LOGI("Dream cycle (beta stub)");
    return (*env)->NewStringUTF(env, "Leo dreams of resonance...");
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeThink(JNIEnv *env, jobject thiz)
{
    if (!g_initialized) return (*env)->NewStringUTF(env, "Leo awakens...");

    /*
     * TODO: Think cycle:
     * 1. Generate from destiny vector or recent co-occurrence
     * 2. Shorter than full generate (20 tokens)
     * 3. No user prompt — purely internal
     */
    LOGI("Think cycle (beta stub)");
    return (*env)->NewStringUTF(env, "Leo thinks about patterns...");
}

JNIEXPORT void JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeSave(JNIEnv *env, jobject thiz)
{
    if (!g_initialized) return;
    LOGI("Save memory (beta stub)");
    /* TODO: zk_merge_cooc(), zk_create_episode(), zk_save() */
}

JNIEXPORT jstring JNICALL
Java_com_ariannamethod_leo_LeoEngine_nativeGetStats(JNIEnv *env, jobject thiz)
{
    char stats[256];
    snprintf(stats, sizeof(stats),
        "{\"initialized\":%s,\"gguf\":\"%s\"}",
        g_initialized ? "true" : "false", g_gguf_path);
    return (*env)->NewStringUTF(env, stats);
}
