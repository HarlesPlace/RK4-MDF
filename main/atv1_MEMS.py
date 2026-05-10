import numpy as np
import matplotlib.pyplot as plt
import os

# --- Parâmetros (Convertidos para SI) ---
m = 0.005         # kg
k = 40.0          # N/m
c = 0.1           # N.s/m
a = 0.1           # m (10 cm)
F0 = 1.2          # N
w = 2.0           # rad/s
L = 0.05          # H (50 mH)
alpha = 1.0       # N/A
R = 2.0           # Ohms
condicao_inicial = [0.02, 0.0, 0.01, 0.0, 0.0] # Condição inicial: [x(0), x'(0), y(0), y'(0), i(0)]
l0 = [0.08, 0.12] # m (8 cm e 12 cm)
h = [0.01, 0.005, 0.001, 0.0005]  # s

def derivadas(estado, t_atual, l0):
    x1,x2,x3,x4,x5 = estado
    def l1(x,y):
        return np.sqrt((x + a)**2 + y**2)
    def l2(x,y):
        return np.sqrt((x - a)**2 + y**2)
    L1 = l1(x1,x3)
    L2 = l2(x1,x3)
    dx1dt = x2
    dx2dt = (
        ((-k * x1) / m) * (2 - l0 * ((L1 + L2) / (L1 * L2))) 
        - ((k * a * l0) / m) * ((L1 - L2) / (L1 * L2)) 
        - ((c * x2) / m)
    )
    dx3dt = x4
    dx4dt = (
        ((-k * x3) / m) * (2 - l0 * ((L1 + L2) / (L1 * L2)))
        - ((c * x4) / m) - ((alpha * x5) / m) - ((F0 * np.sin(w * t_atual)) / m)
    )
    dx5dt = ((alpha * x4 - R * x5) / L)
    return np.array([dx1dt, dx2dt, dx3dt, dx4dt, dx5dt])

def euler(passo, t_simul, condicao_inicial, l0=l0[0]):
    n_passos = int(t_simul / passo)
    t = np.linspace(0, t_simul, n_passos)
    resultados = np.zeros((n_passos, 5))
    ax = np.zeros(n_passos)
    ay = np.zeros(n_passos)
    didt = np.zeros(n_passos)

    # Condição inicial
    resultados[0] = condicao_inicial

    for i in range(0, n_passos - 1):
        f = derivadas(resultados[i], t[i], l0)
        # --- TRAVA DE SEGURANÇA ---
        # Se qualquer derivada ou valor de estado ficar absurdo, paramos tudo
        if np.any(np.abs(f) > 1e100) or np.any(np.abs(resultados[i]) > 1e100):
            # Preenche o ponto atual e todo o resto com NaN para não quebrar o gráfico
            ax[i:] = np.nan
            ay[i:] = np.nan
            didt[i:] = np.nan
            resultados[i+1:] = np.nan
            break # Sai do loop imediatamente
        ax[i] = f[1]   # aceleração em x
        ay[i] = f[3]   # aceleração em y
        didt[i] = f[4] # variação da corrente
        resultados[i+1] = resultados[i] + passo * f
    return t, resultados, ax, ay, didt

def salvar_grafico_8_subplots(t_lista, dados_lista, accel_x_lista, 
                              accel_y_lista, pot_lista, energia_lista, h_lista, 
                              l0_val, sufixo_nome, prefixo="Euler"):
    """
    Gera uma imagem de 8 subplots (1 coluna) para um L0 específico.
    Pode conter múltiplas curvas (uma para cada h).
    """
    # --- Preparação do Plot ---
    fig, axs = plt.subplots(8, 1, figsize=(10, 20), sharex=True)
    fig.suptitle('Simulação de Coletor de Energia MEMS Biestável\n' + 
                f'Método de {prefixo} - $l_0$ : {l0_val}m \n{sufixo_nome}', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.subplots_adjust(hspace=0.5)

    # Nomes para os labels dos subplots
    labels = ['$y(t)$', '$\dot{y}(t)$', '$\ddot{y}(t)$', 
            '$x(t)$', '$\dot{x}(t)$', '$\ddot{x}(t)$', 
            '$I(t)$', '$P(t)$']
    
    # Plotando cada h fornecido na lista
    for i, h in enumerate(h_lista):
        # Captura a energia correspondente a este h
        e_atual = energia_lista[i]

        # Define o texto da legenda para a energia (trata infinito)
        if np.isnan(e_atual) or np.isinf(e_atual):
            legenda_energia = f'h={h} (E=$\infty$)'
        else:
            legenda_energia = f'h={h} (E={e_atual:.4f} J)'

        axs[0].plot(t_lista[i], dados_lista[i][:,2] , label=f'h={h}')     # y
        axs[1].plot(t_lista[i], dados_lista[i][:,3])                      # vy
        axs[2].plot(t_lista[i], accel_y_lista[i])                         # ay
        axs[3].plot(t_lista[i], dados_lista[i][:,0])                      # x
        axs[4].plot(t_lista[i], dados_lista[i][:,1])                      # vx
        axs[5].plot(t_lista[i], accel_x_lista[i])                         # ax
        axs[6].plot(t_lista[i], dados_lista[i][:,4])                      # I
        axs[7].plot(t_lista[i], pot_lista[i], label=legenda_energia)      # P
   
    # --- Ajustes de Eixo e Escala ---
    idx_melhor = -1
    for i in range(8):
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].set_xlim(0, 20)

        # Fixar escala baseada no melhor resultado (se ele não for NaN)
        # Pegamos os dados do melhor passo para definir os limites
        melhor_resultado_da_linha = []
        if i == 0: melhor_resultado_da_linha = dados_lista[idx_melhor][:,2]   # y
        elif i == 1: melhor_resultado_da_linha = dados_lista[idx_melhor][:,3] # vy
        elif i == 2: melhor_resultado_da_linha = accel_y_lista[idx_melhor]    # ay
        elif i == 3: melhor_resultado_da_linha = dados_lista[idx_melhor][:,0] # x
        elif i == 4: melhor_resultado_da_linha = dados_lista[idx_melhor][:,1] # vx
        elif i == 5: melhor_resultado_da_linha = accel_x_lista[idx_melhor]    # ax
        elif i == 6: melhor_resultado_da_linha = dados_lista[idx_melhor][:,4] # I
        elif i == 7: melhor_resultado_da_linha = pot_lista[idx_melhor]        # P

        # senão for uma lista só de NaNs fixamos a escala
        if np.any(np.isfinite(melhor_resultado_da_linha)):
            y_min = np.nanmin(melhor_resultado_da_linha)
            y_max = np.nanmax(melhor_resultado_da_linha)
            # margem de 10% para não colar na borda
            margem = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
            axs[i].set_ylim(y_min - margem, y_max + margem)

    axs[0].legend(loc='upper right')
    axs[7].legend(loc='upper right')
    axs[7].set_xlabel('Tempo (s)')
    
    # Verifica se a pasta existe; se não, cria automaticamente
    diretorio_saida = "resultados"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")

    # Nome do arquivo (substitui se já existir)
    nome_arquivo = f"{prefixo}_{l0_val}_{sufixo_nome.replace(' ', '_').replace('ú', 'u').replace('ê', 'e')}.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

    # Salva com alta qualidade e sem margens extras
    plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")

    plt.close(fig) # Fecha a figura 

#####################################################################################
def main():
    dados_completos_euler = {l: {"t": [], "res": [], "ax": [], "ay": [], "p": [], "e": []} for l in l0}

    for val_l0 in l0:
        for passo in h:
            t, resultados, ax, ay, didt = euler(passo, 20, condicao_inicial, l0=val_l0)
            x_pos = resultados[:, 0]
            x_vel = resultados[:, 1]
            x_acel = ax
            y_pos = resultados[:, 2]
            y_vel = resultados[:, 3]
            y_acel = ay
            corrente = resultados[:, 4]
            potencia = R * (corrente**2)
            energia_total = np.sum(potencia) * passo
            # Regra do Trapézio (um pouco mais precisa que a soma simples)
            energia_total_trap = np.trapz(potencia, dx=passo)

            # Gerar imagem INDIVIDUAL (1 L0, 1 h)
            salvar_grafico_8_subplots([t], [resultados], [x_acel], [y_acel], [potencia], [energia_total_trap], [passo], val_l0, f"Passo único h={passo}", prefixo="Euler")

            # Guardar para as composições
            dados_completos_euler[val_l0]["t"].append(t)
            dados_completos_euler[val_l0]["res"].append(resultados)
            dados_completos_euler[val_l0]["ax"].append(x_acel)
            dados_completos_euler[val_l0]["ay"].append(y_acel)
            dados_completos_euler[val_l0]["p"].append(potencia)
            dados_completos_euler[val_l0]["e"].append(energia_total_trap)

        # Gerar imagem de CONVERGÊNCIA (1 L0, 4 passos h juntos)
        d = dados_completos_euler[val_l0]
        salvar_grafico_8_subplots(d["t"], d["res"], d["ax"], d["ay"], d["p"], d["e"], h, val_l0, "Convergência de h")

if __name__ == "__main__":
    main()