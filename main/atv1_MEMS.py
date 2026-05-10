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

def derivadas(estado, t_atual, l0, R=2.0, w=2.0):
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

def euler(passo, t_simul, condicao_inicial, l0=l0[0], R=2.0, w=2.0):
    n_passos = int(t_simul / passo)
    t = np.linspace(0, t_simul, n_passos)
    resultados = np.zeros((n_passos, 5))
    ax = np.zeros(n_passos)
    ay = np.zeros(n_passos)
    didt = np.zeros(n_passos)

    # Condição inicial
    resultados[0] = condicao_inicial

    for i in range(0, n_passos - 1):
        f = derivadas(resultados[i], t[i], l0, R, w)

        proximo_estado = resultados[i] + passo * f

        # --- TRAVA DE SEGURANÇA ---
        # Se qualquer derivada ou valor de estado ficar absurdo, paramos tudo
        if np.any(np.abs(f) > 1e100) or np.any(np.abs(proximo_estado) > 1e100):
            # Preenche o ponto atual e todo o resto com NaN para não quebrar o gráfico
            ax[i:] = np.nan
            ay[i:] = np.nan
            didt[i:] = np.nan
            resultados[i+1:] = np.nan
            break # Sai do loop imediatamente

        ax[i] = f[1]   # aceleração em x
        ay[i] = f[3]   # aceleração em y
        didt[i] = f[4] # variação da corrente
        resultados[i+1] = proximo_estado
    return t, resultados, ax, ay, didt

def RK4(passo, t_simul, condicao_inicial, l0=l0[0], R=2.0, w=2.0):
    n_passos = int(t_simul / passo)
    t = np.linspace(0, t_simul, n_passos)
    resultados = np.zeros((n_passos, 5))
    ax = np.zeros(n_passos)
    ay = np.zeros(n_passos)
    didt = np.zeros(n_passos)

    # Condição inicial
    resultados[0] = condicao_inicial

    for i in range(0, n_passos - 1):
        K1 = derivadas(resultados[i], t[i], l0, R, w)
        K2 = derivadas(resultados[i] + passo * K1 / 2, t[i] + passo / 2, l0, R, w)
        K3 = derivadas(resultados[i] + passo * K2 / 2, t[i] + passo / 2, l0, R, w)
        K4 = derivadas(resultados[i] + passo * K3, t[i] + passo, l0, R, w)

        f_media = (K1 + 2*K2 + 2*K3 + K4) / 6
        proximo_estado = resultados[i] + passo * f_media
       
        # --- TRAVA DE SEGURANÇA ---
        # Se qualquer derivada ou valor de estado ficar absurdo, paramos tudo
        if np.any(np.abs(f_media) > 1e100) or np.any(np.abs(proximo_estado) > 1e100):
            # Preenche o ponto atual e todo o resto com NaN para não quebrar o gráfico
            ax[i:] = np.nan
            ay[i:] = np.nan
            didt[i:] = np.nan
            resultados[i+1:] = np.nan
            break # Sai do loop imediatamente

        ax[i] = f_media[1]   # aceleração em x
        ay[i] = f_media[3]   # aceleração em y
        didt[i] = f_media[4] # variação da corrente
        resultados[i+1] = proximo_estado
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
    labels = ['$y(t) [m]$', '$\dot{y}(t) [m/s]$', '$\ddot{y}(t) [m/s^2]$', 
            '$x(t) [m]$', '$\dot{x}(t) [m/s]$', '$\ddot{x}(t) [m/s^2]$', 
            '$I(t) [A]$', '$P(t) [W]$']
    
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
    plt.savefig(caminho_completo, dpi=600, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")

    plt.close(fig) # Fecha a figura 

def salvar_grafico_16_subplots(dados_completos, h_lista, 
                              l0_val, sufixo_nome, prefixo="Euler"):
    """
    Gera uma imagem de 16 subplots (2 colunas) para um L0 específico.
    Pode conter múltiplas curvas (uma para cada h).
    """

    # --- Preparação do Plot ---
    fig, axs = plt.subplots(8, 2, figsize=(16, 22), sharex=True)
    fig.suptitle('Simulação de Coletor de Energia MEMS Biestável\n' + 
                f'Método de {prefixo} \n{sufixo_nome}', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.subplots_adjust(hspace=0.5)

    # Nomes para os labels dos subplots
    labels = ['$y(t) [m]$', '$\dot{y}(t) [m/s]$', '$\ddot{y}(t) [m/s^2]$', 
            '$x(t) [m]$', '$\dot{x}(t) [m/s]$', '$\ddot{x}(t) [m/s^2]$', 
            '$I(t) [A]$', '$P(t) [W]$']
    
    for col, val_l0 in enumerate(l0_val):
        # Adiciona um "Subtítulo" para cada coluna
        d_l0 = dados_completos[val_l0]
        axs[0, col].set_title(f'Configuração: $l_0$ = {val_l0}m', 
                          fontsize=14, pad=20, fontweight='semibold')
        # Plotando cada h fornecido na lista
        for i, h in enumerate(h_lista):
            # Captura a energia correspondente a este h
            t = d_l0["t"][i]
            res = d_l0["res"][i]
            e_atual = d_l0["e"][i]

            # Define o texto da legenda para a energia (trata infinito)
            if np.isnan(e_atual) or np.isinf(e_atual):
                legenda_energia = f'h={h} (E=$\infty$)'
            else:
                legenda_energia = f'h={h} (E={e_atual:.4f} J)'

            axs[0, col].plot(t, res[:,2] , label=f'h={h}')            # y
            axs[1, col].plot(t, res[:,3])                             # vy
            axs[2, col].plot(t, d_l0["ay"][i])                        # ay
            axs[3, col].plot(t, res[:,0])                             # x
            axs[4, col].plot(t, res[:,1])                             # vx
            axs[5, col].plot(t, d_l0["ax"][i])                        # ax
            axs[6, col].plot(t, res[:,4])                             # I
            axs[7, col].plot(t, d_l0["p"][i], label=legenda_energia)  # P
        axs[0, col].legend(loc='upper right')
        axs[7, col].legend(loc='upper right')
        axs[7, col].set_xlabel('Tempo (s)')

    # --- Ajustes de Eixo e Escala ---
    idx_melhor = -1
    for i in range(8):
        for j in range(2):
            if j==0:
                axs[i, 0].set_ylabel(labels[i])
            axs[i, j].grid(True, alpha=0.3)
            axs[i, j].set_xlim(0, 20)

            # Fixar escala baseada no melhor resultado (se ele não for NaN)
            # Pegamos os dados do melhor passo para definir os limites
            melhor_resultado_da_linha = []
            l0_atual = l0_val[j]
            d_l0 = dados_completos[l0_atual]
            if i == 0: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,2]   # y
            elif i == 1: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,3] # vy
            elif i == 2: melhor_resultado_da_linha = d_l0["ay"][idx_melhor]      # ay
            elif i == 3: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,0] # x
            elif i == 4: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,1] # vx
            elif i == 5: melhor_resultado_da_linha = d_l0["ax"][idx_melhor]       # ax
            elif i == 6: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,4] # I
            elif i == 7: melhor_resultado_da_linha = d_l0["p"][idx_melhor]        # P

            # se não for uma lista só de NaNs fixamos a escala
            if np.any(np.isfinite(melhor_resultado_da_linha)):
                y_min = np.nanmin(melhor_resultado_da_linha)
                y_max = np.nanmax(melhor_resultado_da_linha)
                # margem de 10% para não colar na borda
                margem = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                axs[i, j].set_ylim(y_min - margem, y_max + margem)
    # Verifica se a pasta existe; se não, cria automaticamente
    diretorio_saida = "resultados"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")

    # Nome do arquivo (substitui se já existir)
    nome_arquivo = f"{prefixo}_16plots_{sufixo_nome.replace(' ', '_').replace('ú', 'u').replace('ê', 'e').replace('ç', 'c').replace('õ', 'o')}.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

    # Salva com alta qualidade e sem margens extras
    plt.savefig(caminho_completo, dpi=600, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")

    plt.close(fig) # Fecha a figura 

def salvar_grafico_32_subplots(dados_completos_euler, dados_completos_rk4, h_lista, l0_val, sufixo_nome, prefixo="Euler"):

    # --- Preparação do Plot ---
    fig, axs = plt.subplots(8, 4, figsize=(30, 24), sharex=True)
    fig.suptitle(
        "Simulação de Coletor de Energia MEMS Biestável\n"
        "Comparação entre Euler e RK4",
        fontsize=18,
        fontweight='bold',
        y=0.995
    )
    plt.subplots_adjust( hspace=0.5,
                        wspace=0.25)
    colunas = [
        ("Euler", l0_val[0]),
        ("Euler", l0_val[1]),
        ("RK4",   l0_val[0]),
        ("RK4",   l0_val[1]),
    ]

    titulos = [
        f"Euler\n$l_0$ = {l0_val[0]}m",
        f"Euler\n$l_0$ = {l0_val[1]}m",
        f"RK4\n$l_0$ = {l0_val[0]}m",
        f"RK4\n$l_0$ = {l0_val[1]}m",
    ]

    fig.text(0.25, 0.965, "Método de Euler",
         ha='center',
         fontsize=16,
         fontweight='bold')

    fig.text(0.75, 0.965, "Método de RK4",
         ha='center',
         fontsize=16,
         fontweight='bold')
    
    # Nomes para os labels dos subplots
    labels = ['$y(t) [m]$', '$\dot{y}(t) [m/s]$', '$\ddot{y}(t) [m/s^2]$', 
            '$x(t) [m]$', '$\dot{x}(t) [m/s]$', '$\ddot{x}(t) [m/s^2]$', 
            '$I(t) [A]$', '$P(t) [W]$']
    
    for col, (metodo, val_l0_atual) in enumerate(colunas):
        # Escolhe qual dicionário usar
        if metodo == "Euler":
            d_l0 = dados_completos_euler[val_l0_atual]
        else:
            d_l0 = dados_completos_rk4[val_l0_atual]

        # Título da coluna
        axs[0, col].set_title(
            titulos[col],
            fontsize=13,
            pad=20,
            fontweight='semibold'
        )

        for i, h in enumerate(h_lista):

            t = d_l0["t"][i]
            res = d_l0["res"][i]
            e_atual = d_l0["e"][i]

            if np.isnan(e_atual) or np.isinf(e_atual):
                legenda_energia = f'h={h} (E=$\\infty$)'
            else:
                legenda_energia = f'h={h} (E={e_atual:.4f} J)'

            axs[0, col].plot(t, res[:,2], label=f'h={h}')
            axs[1, col].plot(t, res[:,3])
            axs[2, col].plot(t, d_l0["ay"][i])
            axs[3, col].plot(t, res[:,0])
            axs[4, col].plot(t, res[:,1])
            axs[5, col].plot(t, d_l0["ax"][i])
            axs[6, col].plot(t, res[:,4])
            axs[7, col].plot(t, d_l0["p"][i], label=legenda_energia)

        axs[0, col].legend(loc='upper right', fontsize=8)
        axs[7, col].legend(loc='upper right', fontsize=8)

        axs[7, col].set_xlabel('Tempo (s)')

    # --- Ajustes de Eixo e Escala ---
    idx_melhor = -1
    for i in range(8):
        for j in range(4):
            if j==0:
                axs[i, 0].set_ylabel(labels[i])
            axs[i, j].grid(True, alpha=0.3)
            axs[i, j].set_xlim(0, 20)

            # Fixar escala baseada no melhor resultado (se ele não for NaN)
            # Pegamos os dados do melhor passo para definir os limites
            melhor_resultado_da_linha = []
            metodo, l0_atual = colunas[j]
            if metodo == "Euler":
                d_l0 = dados_completos_euler[l0_atual]
            else:
                d_l0 = dados_completos_rk4[l0_atual]
            if i == 0: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,2]   # y
            elif i == 1: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,3] # vy
            elif i == 2: melhor_resultado_da_linha = d_l0["ay"][idx_melhor]      # ay
            elif i == 3: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,0] # x
            elif i == 4: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,1] # vx
            elif i == 5: melhor_resultado_da_linha = d_l0["ax"][idx_melhor]       # ax
            elif i == 6: melhor_resultado_da_linha = d_l0["res"][idx_melhor][:,4] # I
            elif i == 7: melhor_resultado_da_linha = d_l0["p"][idx_melhor]        # P

            # se não for uma lista só de NaNs fixamos a escala
            if np.any(np.isfinite(melhor_resultado_da_linha)):
                y_min = np.nanmin(melhor_resultado_da_linha)
                y_max = np.nanmax(melhor_resultado_da_linha)
                # margem de 10% para não colar na borda
                margem = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                axs[i, j].set_ylim(y_min - margem, y_max + margem)
    
    # Verifica se a pasta existe; se não, cria automaticamente
    diretorio_saida = "resultados"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")

    # Nome do arquivo (substitui se já existir)
    nome_arquivo = f"{prefixo}_32plots_{sufixo_nome.replace(' ', '_').replace('ú', 'u').replace('ê', 'e').replace('ç', 'c').replace('õ', 'o').replace('ã', 'a')}.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

    # Salva com alta qualidade e sem margens extras
    plt.savefig(caminho_completo, dpi=600, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")

    plt.close(fig) # Fecha a figura 

def atv1_1():
    dados_completos_euler = {l: {"t": [], "res": [], "ax": [], "ay": [], "p": [], "e": []} for l in l0}
    dados_completos_rk4 = {l: {"t": [], "res": [], "ax": [], "ay": [], "p": [], "e": []} for l in l0}

    for val_l0 in l0:
        for passo in h:
            # --- MÉTODO DE EULER ---
            t_euler, resultados_euler, ax_euler, ay_euler, didt_euler = euler(passo, 20, condicao_inicial, l0=val_l0, R=R, w=w)
            x_acel_euler = ax_euler
            y_acel_euler = ay_euler
            corrente_euler = resultados_euler[:, 4]
            potencia_euler = R * (corrente_euler**2)
            energia_total_euler = np.sum(potencia_euler) * passo
            # Regra do Trapézio (um pouco mais precisa que a soma simples)
            energia_total_trap_euler = np.trapz(potencia_euler, dx=passo)

            # Gerar imagem INDIVIDUAL (1 L0, 1 h)
            salvar_grafico_8_subplots([t_euler], [resultados_euler], [x_acel_euler], [y_acel_euler],
                                      [potencia_euler], [energia_total_trap_euler], [passo], val_l0, 
                                      f"Passo único h={passo}", prefixo="Euler")

            # Guardar para as composições
            dados_completos_euler[val_l0]["t"].append(t_euler)
            dados_completos_euler[val_l0]["res"].append(resultados_euler)
            dados_completos_euler[val_l0]["ax"].append(x_acel_euler)
            dados_completos_euler[val_l0]["ay"].append(y_acel_euler)
            dados_completos_euler[val_l0]["p"].append(potencia_euler)
            dados_completos_euler[val_l0]["e"].append(energia_total_trap_euler)

            # --- MÉTODO DE RK4 ---
            t_rk4, resultados_rk4, ax_rk4, ay_rk4, didt_rk4 = RK4(passo, 20, condicao_inicial, l0=val_l0, R=R, w=w)
            x_acel_rk4 = ax_rk4
            y_acel_rk4 = ay_rk4
            corrente_rk4 = resultados_rk4[:, 4]
            potencia_rk4 = R * (corrente_rk4**2)
            energia_total_rk4 = np.sum(potencia_rk4) * passo
            # Regra do Trapézio (um pouco mais precisa que a soma simples)
            energia_total_trap_rk4 = np.trapz(potencia_rk4, dx=passo)

            # Gerar imagem INDIVIDUAL (1 L0, 1 h)
            salvar_grafico_8_subplots([t_rk4], [resultados_rk4], [x_acel_rk4], [y_acel_rk4],
                                      [potencia_rk4], [energia_total_trap_rk4], [passo], val_l0, 
                                      f"Passo único h={passo}", prefixo="RK4")

            # Guardar para as composições
            dados_completos_rk4[val_l0]["t"].append(t_rk4)
            dados_completos_rk4[val_l0]["res"].append(resultados_rk4)
            dados_completos_rk4[val_l0]["ax"].append(x_acel_rk4)
            dados_completos_rk4[val_l0]["ay"].append(y_acel_rk4)
            dados_completos_rk4[val_l0]["p"].append(potencia_rk4)
            dados_completos_rk4[val_l0]["e"].append(energia_total_trap_rk4)

        # Gerar imagem de CONVERGÊNCIA (1 L0, 4 passos h juntos)
        # --- EULER ---
        d_euler = dados_completos_euler[val_l0]
        salvar_grafico_8_subplots(d_euler["t"], d_euler["res"], d_euler["ax"], d_euler["ay"], 
                                  d_euler["p"], d_euler["e"], h, val_l0, "Convergência de h", prefixo="Euler")
        # --- RK4 ---
        d_rk4 = dados_completos_rk4[val_l0]
        salvar_grafico_8_subplots(d_rk4["t"], d_rk4["res"], d_rk4["ax"], d_rk4["ay"], 
                                  d_rk4["p"], d_rk4["e"], h, val_l0, "Convergência de h", prefixo="RK4")
        
    # --- EULER ---
    salvar_grafico_16_subplots(dados_completos_euler, h, l0, "Múltiplas Configurações L0", prefixo="Euler")
    # --- RK4 ---
    salvar_grafico_16_subplots(dados_completos_rk4, h, l0, "Múltiplas Configurações L0", prefixo="RK4")
    
    salvar_grafico_32_subplots(dados_completos_euler, dados_completos_rk4, h, l0, "Comparação Completa Euler vs RK4", prefixo="Comparacao")

def atv1_2():
    # --- Otimização de R ---
    resistencias = np.linspace(0.01, 5, 120)
    energias_coletadas_r = []

    for R_atual in resistencias:
        print(f"Simulando para R = {R_atual:.4f} Ohms...")
        t, res, ax, ay, didt = RK4(passo=0.001, t_simul=20, condicao_inicial=condicao_inicial, l0=0.12, R=R_atual, w=w)
        corrente = res[:, 4]
        potencia = R_atual * (corrente**2)
        energia = np.trapz(potencia, x=t)
        print(f"  Energia coletada: {energia:.6f} J")
        energias_coletadas_r.append(energia)
    
    # --- Plotagem R vs Energia ---
    plt.figure(figsize=(10, 6))
    plt.plot(resistencias, energias_coletadas_r, 'b-', linewidth=2)
    plt.xlabel('Resistência $R$ [$\Omega$]')
    plt.ylabel('Energia Total Coletada [J]')
    plt.title('Otimização da Resistência de Carga')
    plt.grid(True, alpha=0.3)

    # Encontra o índice onde a energia é máxima
    idx_max = np.argmax(energias_coletadas_r)

    # Pega o valor de R e a energia correspondente
    R_otimo = resistencias[idx_max]
    E_maxima = energias_coletadas_r[idx_max]

    print(f"O valor de R que maximiza a energia é: {R_otimo:.4f} Ohms")
    print(f"Energia máxima coletada: {E_maxima:.6f} J")

    # Marcar o ponto no gráfico
    plt.scatter(R_otimo, E_maxima, color='red', s=80, zorder=5)
    plt.annotate(f'Máximo:\n$R = {R_otimo:.2f} \Omega$\n$E = {E_maxima:.6f} J$',
             xy=(R_otimo, E_maxima), 
             xytext=(R_otimo + 0.5, E_maxima * 0.9), 
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.legend()

    # Verifica se a pasta existe; se não, cria automaticamente
    diretorio_saida = "resultados"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")

    # Nome do arquivo (substitui se já existir)
    nome_arquivo = f"Otimizacao_Resistencia.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

    # Salva com alta qualidade e sem margens extras
    plt.savefig(caminho_completo, dpi=600, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")
    plt.close() # Fecha a figura 

    #################################################################
    # --- Otimização de W ---
    frequencias = np.linspace(0.01, 40, 120)
    energias_coletadas_w = []

    for w_atual in frequencias:
        print(f"Simulando para w = {w_atual:.4f} rads/s...")
        t, res, ax, ay, didt = RK4(passo=0.001, t_simul=20, condicao_inicial=condicao_inicial, l0=0.12, R=R_otimo, w=w_atual)
        corrente = res[:, 4]
        potencia = R_otimo * (corrente**2)
        energia = np.trapz(potencia, x=t)
        print(f"  Energia coletada: {energia:.6f} J")
        
        energias_coletadas_w.append(energia)

    # --- Plotagem W vs Energia ---
    plt.figure(figsize=(10, 6))
    plt.plot(frequencias, energias_coletadas_w, 'g-', linewidth=2)
    plt.xlabel('Frequência $\\omega$ [rad/s]')
    plt.ylabel('Energia Total Coletada [J]')
    plt.title('Otimização da Frequência de Operação')
    plt.grid(True, alpha=0.3)

    # Encontra o índice onde a energia é máxima
    idx_max = np.argmax(energias_coletadas_w)

    # Pega o valor de w e a energia correspondente
    w_otimo = frequencias[idx_max]
    E_maxima = energias_coletadas_w[idx_max]

    print(f"O valor de w que maximiza a energia é: {w_otimo:.4f} rad/s")
    print(f"Energia máxima coletada: {E_maxima:.6f} J")

    # Marcar o ponto no gráfico
    plt.scatter(w_otimo, E_maxima, color='red', s=80, zorder=5)
    plt.annotate(f'Máximo:\n$\\omega = {w_otimo:.2f} \\frac{{rad}}{{s}}$\n$E = {E_maxima:.6f} J$',
             xy=(w_otimo, E_maxima), 
             xytext=(w_otimo + 0.5, E_maxima * 0.9),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.legend()

    # Verifica se a pasta existe; se não, cria automaticamente
    diretorio_saida = "resultados"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")

    # Nome do arquivo (substitui se já existir)
    nome_arquivo = f"Otimizacao_Frequencia.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

    # Salva com alta qualidade e sem margens extras
    plt.savefig(caminho_completo, dpi=600, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")
    plt.close() # Fecha a figura 

    #################################################################
    # --- Otimização de conjunta R e W ---
    vec_R = np.linspace(0.01, 20, 120)
    vec_W = np.linspace(0.01, 40, 120)

    # Cria as matrizes 2D
    R_mesh, W_mesh = np.meshgrid(vec_R, vec_W)
    Energia_mesh = np.zeros((120, 120))

    for i in range(120):
        for j in range(120):
            print(f"Progresso {i+1}/120: {100*(j+1)/120:.1f}%")
            R_at = R_mesh[i, j]
            W_at = W_mesh[i, j]
            
            # Simulação
            t, res, _, _, _ = RK4(passo=0.001, t_simul=20, condicao_inicial=condicao_inicial, 
                                l0=0.12, R=R_at, w=W_at)
            
            # Cálculo da energia
            corrente = res[:, 4]
            potencia = R_at * (corrente**2)
            Energia_mesh[i, j] = np.trapz(potencia, x=t)
        print(f"Progresso: {100*(i+1)/120:.1f}%")
    
    plt.figure(figsize=(10, 8))

    # Preenchimento das cores (Curvas de Nível)
    cp = plt.contourf(R_mesh, W_mesh, Energia_mesh, levels=20, cmap='viridis')
    plt.colorbar(cp, label='Energia Total [J]')

    # Linhas de contorno para destacar a forma
    plt.contour(R_mesh, W_mesh, Energia_mesh, levels=10, colors='black', alpha=0.3)

    # Localizando o Máximo Global
    idx_max = np.unravel_index(np.argmax(Energia_mesh), Energia_mesh.shape)
    R_best = R_mesh[idx_max]
    W_best = W_mesh[idx_max]
    E_best = Energia_mesh[idx_max]

    # Marcando o ponto ótimo
    plt.scatter(R_best, W_best, color='red', marker='x', s=100, label='Máximo Global')
    plt.annotate(f'Ótimo:\n$R={R_best:.2f}\Omega$\n$\omega={W_best:.2f} rad/s$', 
                xy=(R_best, W_best), xytext=(R_best+0.5, W_best+2),
                arrowprops=dict(arrowstyle='->', color='white'), color='white', fontweight='bold')

    plt.xlabel('Resistência $R$ [$\Omega$]')
    plt.ylabel('Frequência $\omega$ [rad/s]')
    plt.title('Mapa de Colheita de Energia: $R$ vs $\omega$')
    plt.legend()

    # Verifica se a pasta existe; se não, cria automaticamente
    diretorio_saida = "resultados"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")

    # Nome do arquivo (substitui se já existir)
    nome_arquivo = f"Mapa_Otimizacao_Conjunta.png"
    caminho_completo = os.path.join(diretorio_saida, nome_arquivo)

    # Salva com alta qualidade e sem margens extras
    plt.savefig(caminho_completo, dpi=600, bbox_inches='tight')
    print(f"Figura salva em: {caminho_completo}")

    plt.close() # Fecha a figura 


def main():
    #atv1_1()
    atv1_2()

if __name__ == "__main__":
    main()
